#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🪸 Coral LoRA & 大模型 拉取器 —— tkinter 桌面版（作者：@Ne）

零额外依赖（Python 标准库 + 系统自带 tkinter）；装了 Pillow 时封面图显示更佳（没有也能跑，PNG 封面仍可显示）。

双击 启动CoralLoRA.bat 直接弹桌面窗口，不用开浏览器。

功能（轻量，自用够用）：
  1. 搜索 —— C站模型，支持 LoRA / 大模型(Checkpoint) / Embedding / VAE / ControlNet，关键词 + 类型 + 排序
  2. 下载 —— 一键下载本体（实时进度），按类型分目录落盘（models/Lora、models/Stable-diffusion…，可直接给 ComfyUI 用）
  3. 元数据 —— 同时存 JSON（名字/触发词/描述/作者/版本/示例图 URL/versionId）
  4. 标记 —— 搜索结果直接标「✓ 已下载」「🔄 有更新」（本地版本 ≠ C站最新版本）
  5. 已下载 —— 本地列表，按目录分组

配置：
  CORAL_LORA_DIR  下载/扫描目录（默认自动指向本机 ComfyUI models 根，设置页可换并写入 config.txt）
  CORAL_LORA_KEY  Civitai API key（等价于设置页填写，存 下载目录/api_key.txt）
  CORAL_LORA_API  API 地址（默认官方，测试用）
"""
import json
import os
import queue
import re
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:  # Pillow 可选：有则 jpeg/webp 封面都能显示并缩放，没有则只显示 PNG 封面
    from PIL import Image, ImageTk
    HAVE_PIL = True
except Exception:  # noqa: BLE001
    HAVE_PIL = False

# ---------- 配置 ----------
HOME = Path.home()
CONFIG_FILE = Path(__file__).resolve().parent / "config.txt"  # 记住设置页选的目录/API源
UA = "Mozilla/5.0 (CoralLoRA-tk/0.2; +https://github.com/)"
TIMEOUT = 30

API_SOURCES = [("官方 civitai.com（需梯子）", "https://civitai.com/api/v1"),
               ("国内镜像 civitai.red（直连）", "https://civitai.red/api/v1")]

PAGE_TARGET = 12      # 每页目标条数
MAX_FILL_ROUNDS = 3   # 不足时最多补拉轮数


def _config_get(key: str) -> str:
    if not CONFIG_FILE.exists():
        return ""
    try:
        for line in CONFIG_FILE.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                v = line.split("=", 1)[1].strip()
                if v:
                    return v
    except Exception:  # noqa: BLE001
        pass
    return ""


def _config_set(key: str, val: str) -> None:
    try:
        lines = CONFIG_FILE.read_text(encoding="utf-8-sig").splitlines() if CONFIG_FILE.exists() else []
        out, found = [], False
        for ln in lines:
            if ln.strip().startswith(key + "="):
                out.append(f"{key}={val}")
                found = True
            else:
                out.append(ln)
        if not found:
            out.append(f"{key}={val}")
        CONFIG_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def load_config_dir() -> str:
    return _config_get("CORAL_LORA_DIR")


def save_config_dir(path: str) -> None:
    _config_set("CORAL_LORA_DIR", path)


def load_config_api() -> str:
    return _config_get("CORAL_LORA_API")


def save_config_api(api: str) -> None:
    _config_set("CORAL_LORA_API", api)


# 每类模型的下载子目录（相对 DL_DIR）：默认 ComfyUI 标准目录名，
# 可在设置页「下载目录映射」里改，持久化到 config.txt；环境变量 CORAL_LORA_SUBDIR_* 优先级最高。
SUBDIR_KEYS = {
    "LoRA": "CORAL_LORA_SUBDIR_LORA",
    "Checkpoint": "CORAL_LORA_SUBDIR_CHECKPOINT",
    "VAE": "CORAL_LORA_SUBDIR_VAE",
    "Embedding": "CORAL_LORA_SUBDIR_EMBEDDING",
    "ControlNet": "CORAL_LORA_SUBDIR_CONTROLNET",
    "Other": "CORAL_LORA_SUBDIR_OTHER",
}
DEFAULT_SUBDIRS = {
    "LoRA": "loras",
    "Checkpoint": "checkpoints",
    "VAE": "vae",
    "Embedding": "embeddings",
    "ControlNet": "controlnet",
    "Other": "other",
}


def load_subdir_map() -> dict:
    """类别 -> 下载子目录名（相对 DL_DIR）。优先级：环境变量 > config.txt > 默认。"""
    out = dict(DEFAULT_SUBDIRS)
    for cat, key in SUBDIR_KEYS.items():
        v = (os.environ.get(key) or _config_get(key) or "").strip().strip("/\\")
        if v:
            out[cat] = v
    return out


def save_subdir_map(m: dict) -> None:
    for cat, key in SUBDIR_KEYS.items():
        v = (m.get(cat) or "").strip().strip("/\\")
        _config_set(key, v if v else DEFAULT_SUBDIRS[cat])


# 图片代理：原版 CDN image.civitai.com 国内直连不稳（概率加载不出预览图），
# 抓封面时先试原地址、失败自动走代理兜底。{url} 是原图地址占位，填 off 关闭代理。
IMG_PROXY_KEY = "CORAL_LORA_IMG_PROXY"
DEFAULT_IMG_PROXY = "https://images.weserv.nl/?url={url}"  # 免费图片代理（Cloudflare 线路）


def load_img_proxy() -> str:
    """图片代理模板。优先级：环境变量 > config.txt；空 = 用默认 weserv。"""
    return (os.environ.get(IMG_PROXY_KEY) or _config_get(IMG_PROXY_KEY) or "").strip()


def save_img_proxy(v: str) -> None:
    _config_set(IMG_PROXY_KEY, v.strip())


def img_proxy_url(url: str) -> str | None:
    """把图片 URL 套进代理模板；配置为 off/none/- 或代理不可用时返回 None。"""
    tpl = load_img_proxy() or DEFAULT_IMG_PROXY
    if tpl.lower() in ("off", "none", "-"):
        return None
    q = urllib.parse.quote(url, safe="")
    return tpl.replace("{url}", q) if "{url}" in tpl else tpl.rstrip("/") + "?url=" + q


def cover_candidates(url: str) -> list:
    """封面候选抓取地址：原 CDN 优先，代理兜底（去重）。"""
    small = cover_url_small(url)
    cands = [small]
    proxy = img_proxy_url(small)
    if proxy and proxy not in cands:
        cands.append(proxy)
    return cands


# 封面磁盘缓存上限（MB）：超限自动按最旧优先滚动清除。优先级：环境变量 > config.txt > 默认 200MB。
CACHE_MB_KEY = "CORAL_LORA_COVER_CACHE_MB"
DEFAULT_CACHE_MB = 200
_cache_total = 0      # 进程内累计缓存字节（避免每次写都全量扫描）
_cache_scanned = False  # 是否已做过首次全量统计


def cover_cache_limit_bytes() -> int:
    v = (os.environ.get(CACHE_MB_KEY) or _config_get(CACHE_MB_KEY) or "").strip()
    try:
        return max(10, int(float(v) * 1024 * 1024)) if v else DEFAULT_CACHE_MB * 1024 * 1024
    except Exception:  # noqa: BLE001
        return DEFAULT_CACHE_MB * 1024 * 1024


def _evict_cover_cache() -> None:
    """缓存超限就删最旧的文件，直到总量 <= 上限。首次写入时做全量扫描。"""
    global _cache_total, _cache_scanned  # noqa: PLW0603
    d = META_DIR / "covers"
    if not d.is_dir():
        _cache_total = 0
        return
    limit = cover_cache_limit_bytes()
    if _cache_scanned and _cache_total <= limit:
        return
    # 全量扫描（首次或上一轮删过之后，重新数）
    files, total = [], 0
    for f in d.iterdir():
        if f.is_file():
            try:
                st = f.stat()
            except OSError:
                continue
            total += st.st_size
            files.append((st.st_mtime, st.st_size, f))
    _cache_total, _cache_scanned = total, True
    if total <= limit:
        return
    files.sort(key=lambda x: x[0])  # 最旧在前，先删旧的
    freed = 0
    for _, size, f in files:
        if total - freed <= limit:
            break
        try:
            f.unlink()
            freed += size
        except OSError:
            pass
    _cache_total = total - freed


def cover_fetch_bytes(cache_key, url: str) -> bytes | None:
    """带磁盘缓存的封面抓取：缓存命中直接读本地；否则依次试候选 URL，成功写缓存。
    已配置 API key 时带上鉴权头（部分 NSFW/登录图 403 就是缺这个）。
    缓存超上限时自动按最旧优先滚动清除（默认 200MB，CORAL_LORA_COVER_CACHE_MB 可调）。"""
    global _cache_total  # noqa: PLW0603
    cache = (META_DIR / "covers") / (safe_name(str(cache_key)) + ".png")
    if cache.exists():
        try:
            return cache.read_bytes()
        except Exception:  # noqa: BLE001
            pass
    hdrs = {}
    key = api_key()
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    for u in cover_candidates(url):
        try:
            data = http_get_bytes(u, headers=hdrs, timeout=8)
            if data:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_bytes(data)
                    _cache_total += len(data)
                    _evict_cover_cache()
                except Exception:  # noqa: BLE001
                    pass
                return data
        except Exception:  # noqa: BLE001
            continue
    return None


CIVITAI_API = (os.environ.get("CORAL_LORA_API") or load_config_api() or API_SOURCES[0][1]).rstrip("/")


def _auto_dir() -> Path:
    """默认目录（通用启发，不写死机器路径）：当前目录/用户目录下常见的 ComfyUI models，
    都没有则退回下载目录。个人实际目录请用设置页选并持久化到 config.txt（不入库）。"""
    for p in (Path.cwd() / "ComfyUI" / "models", Path.cwd() / "models", HOME / "ComfyUI" / "models"):
        if p.exists():
            return p
    return HOME / "Downloads" / "CoralLoRA"


_cfg_dir = os.environ.get("CORAL_LORA_DIR") or load_config_dir()
DL_DIR = Path(_cfg_dir) if _cfg_dir else _auto_dir()
META_DIR = DL_DIR / "_meta"

# 类型 -> ComfyUI models 根下的子目录名（下载落盘 + 已下载扫描都看这里）
TYPE_DIRS = {
    "LORA": "loras",
    "Checkpoint": "checkpoints",
    "Embedding": "embeddings",
    "TextualInversion": "embeddings",
    "VAE": "vae",
    "Controlnet": "controlnet",
    "LyCORIS": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "IPAdapter": "ipadapter",
    "UNET": "unet",
    "DiffusionModel": "diffusion_models",
    "MotionModule": "motion",
    "Poses": "poses",
    "Wildcards": "wildcards",
    "Workflows": "workflows",
    "Upscaler": "upscale_models",
    "Other": "other",
}
DEFAULT_TYPES = "LORA,CHECKPOINT"

# 界面下拉项：显示名 -> API types 值列表（[] = 不限制）
# 注意：Civitai v1 API 不支持逗号合并 types（Embedding/LyCORIS 也不是合法枚举），
#       多类型会逐类型请求后按 id 去重合并
TYPES = [
    ("LoRA + 大模型", ["LORA", "Checkpoint"]),
    ("仅 LoRA", ["LORA"]),
    ("仅大模型", ["Checkpoint"]),
    ("Embedding", ["TextualInversion"]),
    ("VAE", ["VAE"]),
    ("ControlNet", ["Controlnet"]),
    ("全部类型", []),
]
SORTS = ["Most Downloaded", "Newest", "Highest Rated", "Most Discussed"]

# 用途：存储用英文规范值（目录/筛选一致），中文名按 C站官方标签分类（civitai.red 词典）
KIND_PRESETS = ["style", "concept", "character", "clothing", "pose", "objects", "background", "action"]
KIND_ZH = {"风格": "style", "概念": "concept", "角色": "character", "服装": "clothing",
           "姿势": "pose", "物体": "objects", "背景": "background", "动作": "action"}
KIND_EN2ZH = {v: k for k, v in KIND_ZH.items()}
KIND_LABELS = sorted(KIND_ZH) + ["其他"]


# 类别：显示中文（大模型=Checkpoint），存储用英文规范值
CAT_ZH = {"大模型": "Checkpoint"}
CAT_EN2ZH = {"Checkpoint": "大模型"}
CAT_LABELS = ["LoRA", "大模型", "VAE", "Embedding", "ControlNet", "其他"]


def cat_canon(s: str) -> str:
    s = (s or "").strip()
    return CAT_ZH.get(s, s)


def cat_display(s: str) -> str:
    s = (s or "").strip()
    return CAT_EN2ZH.get(s, s)


def kind_canon(s: str) -> str:
    """中文用途 -> 英文规范值（风格→style…）；其他原样保留。"""
    s = (s or "").strip()
    return KIND_ZH.get(s, s)


def kind_display(s: str) -> str:
    """英文规范值 -> 中文展示（style→风格…）。"""
    s = (s or "").strip()
    return KIND_EN2ZH.get(s, s)

# 配色（GitHub Dark 风格：高对比，文字不会被底色盖住）
BG, PANEL, PANEL2 = "#0d1117", "#161b22", "#21262d"
FG, MUTED = "#e6edf3", "#8b949e"
ACCENT, ACCENT2, BORDER = "#1f6feb", "#58a6ff", "#30363d"
OK, WARN, ERR = "#3fb950", "#d29922", "#f85149"

DL_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 工具 ----------
def api_key() -> str:
    """优先级：环境变量 > 当前目录 key 文件 > 旧版默认目录 key 文件。utf-8-sig 兼容带 BOM 的文本。"""
    env = os.environ.get("CORAL_LORA_KEY", "").strip()
    if env:
        return env
    for keyfile in (DL_DIR / "api_key.txt", HOME / "Downloads" / "CoralLoRA" / "api_key.txt"):
        if keyfile.exists():
            return keyfile.read_text(encoding="utf-8-sig").strip()
    return ""


def set_dl_dir(path: Path) -> None:
    """运行时切换下载/扫描目录（设置页用），并持久化到 config.txt。"""
    global DL_DIR, META_DIR, _cache_total, _cache_scanned  # noqa: PLW0603
    DL_DIR = Path(path)
    META_DIR = DL_DIR / "_meta"
    _cache_total, _cache_scanned = 0, False  # 缓存统计跟着新目录走
    DL_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    save_config_dir(str(DL_DIR))


def save_api_key(key: str) -> None:
    keyfile = DL_DIR / "api_key.txt"
    if key:
        keyfile.write_text(key.strip(), encoding="utf-8")
    elif keyfile.exists():
        keyfile.unlink()


def http_get_bytes(url: str, headers: dict | None = None, timeout: int = TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def cover_url_small(url: str) -> str:
    """把 C站封面 URL 换成小尺寸：新格式 original=true 段换 width=320，老格式 width=xxx 改小。"""
    if not url:
        return ""
    if "original=true" in url:
        return url.replace("original=true", "width=320")
    if "width=" in url:
        return re.sub(r"width=\d+", "width=320", url)
    return url


def first_cover(model: dict) -> str | None:
    """取封面图：优先第一版本的图；第一版没图就扫全部版本找第一张有图的（镜像 API 经常不给顶层 images）。"""
    for vv in model.get("modelVersions") or []:
        imgs = vv.get("images") or []
        if imgs and imgs[0].get("url"):
            return imgs[0]["url"]
    imgs = model.get("images") or []
    return imgs[0].get("url") if imgs else None


# ---------- 便签（本地收藏打标/筛选） ----------
TAGS_FILE = META_DIR / "tags.json"
CAT_MAP = {"LORA": "LoRA", "Checkpoint": "Checkpoint", "VAE": "VAE", "Embedding": "Embedding",
           "TextualInversion": "Embedding", "Controlnet": "ControlNet", "LoCon": "LoRA", "DoRA": "LoRA"}


def load_tags() -> dict:
    """{rel_path: {cat, branch, kind, tags:[...]}}"""
    try:
        return json.loads(TAGS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_tags(tags: dict) -> None:
    try:
        TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        TAGS_FILE.write_text(json.dumps(tags, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def cat_of(meta: dict | None) -> str:
    return CAT_MAP.get((meta or {}).get("type") or "", "其他")


def branch_hint(rel_str: str, meta: dict | None) -> str:
    """分支默认值：优先 baseModel（C站字段），否则用 loras 下的主题子目录名。"""
    b = (meta or {}).get("baseModel")
    if b:
        return b
    parts = Path(rel_str).parts
    if len(parts) >= 2 and parts[0] == "loras":
        return parts[1]
    return ""


def _to_png_bytes(data: bytes):
    """转成 96px 小 PNG 字节（工作线程做，主线程只建 PhotoImage 用）。
    PIL 在时缩放+转码；没有则仅透传 PNG。"""
    if HAVE_PIL:
        try:
            from io import BytesIO
            img = Image.open(BytesIO(data))
            img = img.convert("RGB")
            img.thumbnail((96, 96))
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            pass
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data
    return None


def all_tag_keywords() -> list:
    """便签里所有用过的关键词（筛选下拉直接用，不用手输）。"""
    seen = set()
    for rec in load_tags().values():
        for t in rec.get("tags") or []:
            seen.add(t)
    return sorted(seen)


def all_tag_kinds() -> list:
    """便签里所有用过的用途 + 常用预设。"""
    seen = set(KIND_PRESETS)
    for rec in load_tags().values():
        if rec.get("kind"):
            seen.add(rec["kind"])
    return sorted(seen)


def download_url_for(vid) -> str:
    """当前 API 源对应的模型下载地址（com / red 各自 /api/download/models/）。"""
    base = CIVITAI_API
    if base.endswith("/api/v1"):
        base = base[: -len("/api/v1")]
    return f"{base}/api/download/models/{vid}"


# C站原生基座（大模型）标签：API 探测确认的合法 baseModels 值（Anima / Flux / Illustrious / Pony…）
BASE_MODELS = [
    "SD 1.4", "SD 1.5", "SD 2.0", "SD 2.1", "SDXL 0.9", "SDXL 1.0",
    "SD 3", "SD 3.5", "Pony", "Illustrious", "NoobAI",
    "Flux", "Flux.1 D", "Flux.1 S", "Flux.1 K", "Flux.1 G", "Flux.1 L",
    "Anima", "Qwen", "Hunyuan", "Wan", "Cascade", "SVD", "Playground",
    "PixArt", "Kolors", "Kandinsky", "Openjourney", "Mistoon", "AuraFlow",
    "DiT", "GLM", "SeaArt", "Zeek",
]


def known_branches() -> list:
    """分支下拉：C站原生基座 + 你 loras 下的主题子目录 + 便签里用过的分支。"""
    names = set(BASE_MODELS)
    for rec in load_tags().values():
        if rec.get("branch"):
            names.add(rec["branch"])
    loras_dir = DL_DIR / "loras"
    if loras_dir.is_dir():
        for sub in loras_dir.iterdir():
            if sub.is_dir():
                names.add(sub.name)
    names.add("Other")
    return sorted(names)


def civitai_get(path: str, params: dict | None = None, headers: dict | None = None) -> dict:
    """GET Civitai API，返回 JSON；异常抛 ValueError。已配置 key 时自动带鉴权。
    503/502/504/429（服务器过载）自动重试，间隔递增。"""
    url = f"{CIVITAI_API}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    hdrs = {"User-Agent": UA, "Accept": "application/json", **(headers or {})}
    key = api_key()
    if key and "Authorization" not in hdrs:
        hdrs["Authorization"] = f"Bearer {key}"
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in RETRY_CODES and i < 3:
                time.sleep(2 * (i + 1))
                continue
            raise
        except Exception:  # noqa: BLE001 连接级失败也重试一轮
            if i < 3:
                time.sleep(2 * (i + 1))
                continue
            raise


class _StripAuthRedirect(urllib.request.HTTPRedirectHandler):
    """下载重定向到 CDN/S3（b2 / R2 / cloudflare）时剥掉 Authorization 头：
    urllib 会把首跳的鉴权头原样带到重定向请求，S3/R2 后端不认 → 400 Missing x-amz-content-sha256。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            for k in [k for k in list(new.headers) if k.lower() == "authorization"]:
                del new.headers[k]
        return new


RETRY_CODES = (502, 503, 504, 429)  # 暂时性错误：服务器过载/限流，值得重试


def alt_download_url(url: str) -> str | None:
    """当前源的下载地址失败时换另一个源（civitai.com <-> civitai.red）再试一轮。"""
    for a, b in (("civitai.red", "civitai.com"), ("civitai.com", "civitai.red")):
        if a in url:
            return url.replace(a, b)
    return None


def _open_download(url: str, headers: dict):
    """打开下载流：503/502/504/429 自动重试（间隔递增，最多 3 次）；
    当前源仍失败则换备用源（com<->red）再试一轮；连接失败也直接换源。"""
    opener = urllib.request.build_opener(_StripAuthRedirect)
    urls = [url]
    alt = alt_download_url(url)
    if alt and alt != url:
        urls.append(alt)
    last: Exception | None = None
    for u in urls:
        for i in range(4):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": UA, **headers})
                return opener.open(req, timeout=60)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in RETRY_CODES and i < 3:
                    time.sleep(2 * (i + 1))
                    continue
                break  # 非暂时性错误或重试耗尽 → 换下一个源
            except Exception as e:  # noqa: BLE001 连接/超时 → 换下一个源
                last = e
                break
    raise last


def dl_error_text(exc: Exception) -> str:
    """把下载异常翻译成看得懂的话（503 是服务器暂时过载，403 是权限问题）。"""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 503:
            return "C站/CDN 暂时过载(503)，自动重试+换源仍失败，请稍后再试"
        if exc.code == 403:
            return "需要权限(403)：该模型需登录/API key（设置页填 key，或换 civitai.com 源）"
        if exc.code == 401:
            return "未登录(401)：设置页填 API key 后重试"
        if exc.code == 400:
            return f"请求被拒(400)：{getattr(exc, 'reason', '')}"
        return f"HTTP {exc.code}"
    return str(exc) or type(exc).__name__


def civitai_download(url: str, dest: Path, dl_id: str, headers: dict, q: queue.Queue) -> None:
    """流式下载到 dest（先 .part 后原子改名），进度经 q 上报给 GUI。
    503/502/504/429 自动重试 + 换源兜底；重定向时自动去 Authorization（否则 CDN/S3 后端 400）。"""
    try:
        with _open_download(url, headers) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            q.put(("dl", dl_id, {"file": str(dest), "total": total, "done": 0, "status": "downloading"}))
            tmp = dest.with_suffix(dest.suffix + ".part")
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if done % (1024 * 1024) < 262144 or done == total:
                        q.put(("dl", dl_id, {"file": str(dest), "total": total, "done": done, "status": "downloading"}))
            tmp.replace(dest)
            q.put(("dl", dl_id, {"file": str(dest), "total": total, "done": done, "status": "done"}))
    except Exception as exc:  # noqa: BLE001
        q.put(("dl", dl_id, {"file": str(dest), "total": 0, "done": 0, "status": "error: " + dl_error_text(exc)}))


def safe_name(name: str) -> str:
    """文件名净化：去掉 Windows 非法字符。"""
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return name[:120] or "model"


def meta_path_for(rel: Path) -> Path:
    """按文件相对下载目录的路径映射元数据 JSON（子目录用 __ 拼名，避免同名冲突）。"""
    return META_DIR / ("__".join(rel.with_suffix("").parts) + ".json")


def load_meta(rel: Path) -> dict | None:
    p = meta_path_for(rel)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


_SF_CACHE: dict = {}  # str(文件路径) -> (st_size, st_mtime_ns, 解析出的元数据)


def read_safetensors_meta(path: Path) -> dict:
    """只读 safetensors 头部 JSON（前 8 字节 = 头长度），不读整个文件。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                return {}
            n = int.from_bytes(head, "little")
            if n <= 0 or n > 10 * 1024 * 1024:  # 头部上限 10MB 防呆
                return {}
            data = fh.read(n)
        meta = json.loads(data)
        return meta.get("__metadata__") or {}
    except Exception:  # noqa: BLE001
        return {}


def enrich_from_safetensors(path: Path) -> dict:
    """从 safetensors 训练元数据补 名称/基座/触发词（老 Lora 没有 sidecar 时也能认出）。"""
    md = read_safetensors_meta(path)
    if not md:
        return {}
    out: dict = {}
    name = md.get("ss_output_name") or md.get("modelspec.title")
    if name:
        out["name"] = str(name)
    base = md.get("ss_base_model_version") or md.get("ss_sd_model_name") or md.get("modelspec.architecture")
    if base:
        out["baseModel"] = str(base)
    tf = md.get("ss_tag_frequency")
    if tf:
        try:
            freq = json.loads(tf)
            tags = []
            for _cat, items in freq.items():
                if isinstance(items, dict):
                    tags.extend(list(items.keys()))
            tags = [t for t in tags if t and not t.startswith("@")]  # 去掉 @触发标记
            if tags:
                out["trainedWords"] = tags[:5]
        except Exception:  # noqa: BLE001
            pass
    return out


def sf_enrich_cached(f: Path) -> dict:
    """带缓存地读 safetensors 头部（按 大小+mtime 失效，避免每次刷新重读）。"""
    try:
        st = f.stat()
    except Exception:  # noqa: BLE001
        return {}
    key = str(f)
    cached = _SF_CACHE.get(key)
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
        return cached[2]
    enr = enrich_from_safetensors(f)
    if len(_SF_CACHE) > 2000:
        _SF_CACHE.clear()
    _SF_CACHE[key] = (st.st_size, st.st_mtime_ns, enr)
    return enr


def find_preview(f: Path) -> Path | None:
    """找相邻预览图（ComfyUI-Manager 惯例：<同名>.jpeg/.webp/.png/.preview.png）。"""
    stem = f.stem
    for cand in (f.with_name(stem + ".preview.png"), f.with_name(stem + ".jpeg"),
                 f.with_name(stem + ".jpg"), f.with_name(stem + ".png"), f.with_name(stem + ".webp")):
        if cand.exists():
            return cand
    return None


def load_sidecar_meta(f: Path) -> dict | None:
    """读 C站下载器 / ComfyUI-Manager 写的 <同名>.metadata.json，提取权威 modelId / versionId。

    这样本工具没经手下载的现有模型也能精确匹配「已下载 / 有更新」。"""
    cand = f.with_suffix(".metadata.json")  # xxx.safetensors -> xxx.metadata.json
    if not cand.exists():
        return None
    try:
        d = json.loads(cand.read_text(encoding="utf-8"))
        c = d.get("civitai") or {}
        mid = c.get("modelId") or c.get("model_id")
        if not mid:
            return None
        return {
            "id": mid,
            "versionId": c.get("id") or c.get("versionId") or c.get("version_id"),
            "version": c.get("name"),
            "name": d.get("model_name") or (c.get("model") or {}).get("name"),
            "type": (c.get("model") or {}).get("type") or d.get("sub_type"),
            "baseModel": d.get("base_model") or (c.get("model") or {}).get("baseModel"),
            "trainedWords": c.get("trainedWords") or [],
            "cover": d.get("preview_url") or ((c.get("images") or [{}])[0].get("url") if c.get("images") else None),
            "fromSidecar": True,
        }
    except Exception:  # noqa: BLE001
        return None


def type_subdir(mtype: str) -> str:
    """模型类型 -> ComfyUI models 根下的子目录名（先精确匹配，再试大写）。"""
    t = mtype or ""
    return TYPE_DIRS.get(t) or TYPE_DIRS.get(t.upper()) or TYPE_DIRS.get("Other", "other")


def model_meta(model: dict) -> dict:
    """从 API 模型对象提取关心的元数据（含全部版本，供多基座选择 + 更新检测）。"""
    versions = model.get("modelVersions") or []
    v = versions[0] if versions else {}
    files = v.get("files") or []
    main_file = None
    for f in files:
        if f.get("type") == "Model" or f.get("name", "").endswith((".safetensors", ".ckpt")):
            main_file = f
            break
    if main_file is None and files:
        main_file = files[0]
    trained = v.get("trainedWords") or []
    # 下载地址兜底：列表接口的 downloadUrl 经常为 null，用当前 API 源的版本下载端点补上
    dl_url = (main_file.get("downloadUrl") if main_file else None) \
        or download_url_for(v.get("id"))

    # 全部版本（多基座：同一 LoRA 适配 SDXL/Pony/Illustrious/Flux…）
    vlist = []
    for vv in versions:
        vf = None
        for ff in (vv.get("files") or []):
            if ff.get("type") == "Model" or ff.get("name", "").endswith((".safetensors", ".ckpt")):
                vf = ff
                break
        if vf is None and vv.get("files"):
            vf = vv["files"][0]
        vlist.append({
            "id": vv.get("id"),
            "name": vv.get("name"),
            "baseModel": vv.get("baseModel"),
            "sizeKB": (vf or {}).get("sizeKB"),
            "file": (vf or {}).get("name"),
            "availability": vv.get("availability"),  # Public / EarlyAccess …
            "downloadUrl": (vf or {}).get("downloadUrl") or download_url_for(vv.get("id")),
        })
    base_models = sorted({str(x.get("baseModel")) for x in vlist if x.get("baseModel")})
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "type": model.get("type"),
        "nsfw": model.get("nsfw"),
        "availability": model.get("availability"),  # Public / EarlyAccess（早期访问）
        "creator": (model.get("creator") or {}).get("username"),
        "stats": model.get("stats") or {},
        "description": (model.get("description") or "")[:2000],
        "cover": first_cover(model),
        "version": v.get("name"),
        "versionId": v.get("id"),  # 更新检测用（默认取最新版本）
        "baseModel": v.get("baseModel"),
        "trainedWords": trained,
        "tags": model.get("tags") or [],
        "versions": vlist,          # 多基座选择用
        "baseModels": base_models,
        "file": {
            "name": main_file.get("name") if main_file else None,
            "sizeKB": main_file.get("sizeKB") if main_file else None,
            "downloadUrl": dl_url,
        } if main_file else None,
        "modelUrl": f"https://civitai.com/models/{model.get('id')}",
    }


def local_index() -> tuple[dict, dict]:
    """本地已下载索引：{model_id: {metas: [], files: []}} 与 {文件名小写: [(rel, size), ...]}。"""
    by_id: dict[str, dict] = {}
    by_name: dict[str, list] = {}
    for f in DL_DIR.rglob("*"):
        if f.suffix.lower() not in (".safetensors", ".ckpt"):
            continue
        rel = f.relative_to(DL_DIR)
        meta = load_meta(rel) or load_sidecar_meta(f)
        if meta and meta.get("id"):
            entry = by_id.setdefault(str(meta["id"]), {"metas": [], "files": []})
            entry["metas"].append(meta)
            entry["files"].append(str(rel))
        st = f.stat().st_size
        names = {f.name.lower(), f.stem.lower()}
        if not (meta and meta.get("id")):
            # 老 Lora：safetensors 头里的训练输出名也进索引，扩大「已下载」命中
            enr = sf_enrich_cached(f)
            if enr.get("name"):
                names.add(str(enr["name"]).lower())
        for nm in names:
            by_name.setdefault(nm, []).append((str(rel), st))
    return by_id, by_name


def _vid_key(meta: dict | None) -> int:
    try:
        return int((meta or {}).get("versionId"))
    except (TypeError, ValueError):
        return -1


def annotate_items(items: list[dict]) -> list[dict]:
    """给搜索结果打标记：✓ 已下载（按 model id 精确匹配）/ 🔄 有更新（本地版本 ≠ 最新版本）。
    非本工具下载的同名文件按文件名兜底匹配，标 localMaybe=True。"""
    by_id, by_name = local_index()
    for m in items:
        mid = m.get("id")
        entry = by_id.get(str(mid)) if mid is not None else None
        if entry:
            m["downloaded"] = True
            m["localFiles"] = entry["files"]
            # 多基座：本地版本只要还存在于当前任一版本里就不算过期
            all_vids = {str(x.get("id")) for x in (m.get("versions") or []) if x.get("id")}
            if all_vids:
                same = [mt for mt in entry["metas"]
                        if (mt or {}).get("versionId") and str((mt or {}).get("versionId")) in all_vids]
                if not same:
                    m["updateAvailable"] = True
                    m["localVersion"] = max(entry["metas"], key=_vid_key).get("version")
            continue
        fname = ((m.get("file") or {}).get("name") or "").lower()
        if fname:
            keys = {fname, Path(fname).stem.lower()}  # 全名 + 去扩展名都试
            matches, seen_rel = [], set()
            for k in keys:
                for r, size in by_name.get(k) or []:
                    if r not in seen_rel:
                        seen_rel.add(r)
                        matches.append((r, size))
            if matches:
                m["downloaded"] = True
                m["localFiles"] = [r for r, _ in matches]
                # 大小也在 ±10% 内才算「确信已下载」，否则只是「可能是」
                expected = ((m.get("file") or {}).get("sizeKB") or 0) * 1024
                m["localMaybe"] = not any(
                    expected > 0 and abs(size - expected) / expected < 0.1 for _, size in matches)
    return items


# ---------- GUI 组件 ----------
class ScrollFrame(ttk.Frame):
    """可滚动容器：children 都 pack 进 inner。"""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._refresh_region())
        self.canvas.bind("<Configure>", self._on_canvas_cfg)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")  # 多个滚动区叠加绑定，避免互相覆盖

    def _refresh_region(self, *_):
        # 等布局落定后再取 bbox，避免取到旧尺寸导致滚出大片空白
        self.canvas.after_idle(self._apply_region)

    def _apply_region(self):
        try:
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)
        except tk.TclError:
            pass

    def _on_canvas_cfg(self, e):
        self.canvas.itemconfig(self._win, width=e.width)
        self._refresh_region()

    def _on_wheel(self, e):
        w = self.canvas.winfo_containing(e.x_root, e.y_root)
        if w is None:
            return
        node = w
        while node is not None and node is not self.inner:
            node = getattr(node, "master", None)
        if node is None:
            return
        try:  # 内容不高于视口 → 无可滚动，直接忽略滚轮
            parts = self.canvas.cget("scrollregion").split()
            if len(parts) == 4 and float(parts[3]) <= self.canvas.winfo_height():
                return
        except (tk.TclError, ValueError):
            pass
        self.canvas.yview_scroll(int(-e.delta / 120), "units")
        self._clamp_view()

    def _clamp_view(self):
        """限位兜底：视图不许超出内容范围（防顶部/底部出现空白）。"""
        try:
            top, bottom = self.canvas.yview()
            if top < 0.0:
                self.canvas.yview_moveto(0.0)
            elif bottom > 1.0:
                self.canvas.yview_moveto(max(0.0, 1.0 - (bottom - top)))
        except tk.TclError:
            pass

    def clear(self):
        for child in self.inner.winfo_children():
            child.destroy()
        self.canvas.yview_moveto(0.0)  # 重建内容时回到顶部
        self.canvas.configure(scrollregion=(0, 0, 1, 1))  # 立刻收掉旧滚动区，杜绝短暂空白
        self._refresh_region()


class CoralApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coral LoRA & 大模型 拉取器")
        self.geometry("1020x720")
        self.minsize(880, 580)
        self.configure(bg=BG)
        self.q: queue.Queue = queue.Queue()
        self._covers: dict = {}      # model_id -> thumbnail Label（当前页）
        self._cover_cache: dict = {}  # model_id -> PhotoImage（跨页复用，防膨胀）
        self._ctx = None
        self._pages: list[dict] = []  # 分页缓存 [{items, cursors, has_more}]
        self._page_idx = -1
        self._fetch_token = 0
        self._busy = False
        self._dls: dict = {}         # dl_id -> 状态 dict
        self._local_dirty = True
        self._lfilter: dict = {"cat": "", "branch": "", "kind": "", "kw": "", "dir": ""}
        self._kw_regex = tk.BooleanVar(value=False)  # 关键词正则开关
        self._res_pat = None                        # 搜索页正则（已编译）
        self._res_tag = None                        # 搜索页标签过滤（英文规范值）
        self._cover_sem = threading.BoundedSemaphore(6)  # 封面下载并发上限
        self.show_cover = tk.BooleanVar(value=False)     # 已下载页显示封面开关（大库存建议关）
        self._local_page = 0                             # 已下载列表页码
        self._local_covers: dict = {}                    # rel -> 封面 Label（当前页）
        self._local_cover_cache: dict = {}               # rel -> PhotoImage（缓存防膨胀）
        self._build_styles()
        self._build_ui()
        self.after(200, self._poll)
        self.after(400, self._browse)  # 启动即加载，不搜索也有内容看
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ---------- 样式 ----------
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL,
                    bordercolor=BORDER, lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL)
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Card.TLabel", background=PANEL)
        s.configure("Muted.TLabel", background=BG, foreground=MUTED)
        s.configure("TButton", background=PANEL2, foreground=FG, bordercolor=BORDER, padding=(8, 4))
        s.map("TButton", background=[("active", ACCENT), ("disabled", PANEL)], foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#ffffff")
        s.map("Accent.TButton", background=[("active", ACCENT2), ("disabled", PANEL)])
        s.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=FG, bordercolor=BORDER)
        s.map("TEntry", selectbackground=[("focus", ACCENT)], selectforeground=[("focus", "#ffffff")])
        s.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG, arrowcolor=FG,
                    selectbackground=ACCENT, selectforeground="#ffffff")
        s.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)])
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("Vertical.TScrollbar", background=PANEL2, troughcolor=BG, bordercolor=BORDER, arrowcolor=FG)
        s.configure("TNotebook", background=BG, bordercolor=BORDER)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(12, 6))
        s.map("TNotebook.Tab", background=[("selected", PANEL2)], foreground=[("selected", FG)])
        s.configure("TProgressbar", background=ACCENT2, troughcolor=PANEL, bordercolor=BORDER)
        # 下拉菜单（重点）：列表底色/文字/选中色显式定义，避免文字被底色盖住
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    # ---------- 界面骨架 ----------
    def _build_ui(self):
        # 顶栏
        top = ttk.Frame(self, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Coral LoRA & 大模型 拉取器", font=("", 15, "bold")).pack(side="left")
        self.dir_tag = tk.Label(top, text="", bg=BG, fg=MUTED, font=("", 9))
        self.dir_tag.pack(side="right")
        ttk.Button(top, text="打开目录", command=self._open_dir).pack(side="right", padx=(0, 8))
        self.key_tag = tk.Label(top, text="", bg=BG, fg=MUTED, font=("", 9))
        self.key_tag.pack(side="right", padx=(0, 14))

        # 页签
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=4)
        self._build_search_tab()
        self._build_local_tab()
        self._build_settings_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab)

        # 状态栏
        bar = ttk.Frame(self, padding=(12, 4, 12, 8))
        bar.pack(fill="x")
        self.prog = ttk.Progressbar(bar, maximum=100, value=0)
        self.prog.pack(side="left", fill="x", expand=True)
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(bar, textvariable=self.status_var, bg=BG, fg=MUTED, font=("", 9)).pack(side="left", padx=(12, 0))

        self._refresh_top_tags()

    # ---------- 搜索页 ----------
    def _build_search_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  搜索  ")
        ctl = ttk.Frame(tab, padding=(0, 8, 0, 6))
        ctl.pack(fill="x")

        # 基座下拉：C站原生大模型（baseModels 过滤），类别交给下方分类按钮
        self._search_types = ["LORA", "Checkpoint"]
        self.base_combo = ttk.Combobox(ctl, values=["全部基座"] + BASE_MODELS, state="readonly", width=16)
        self.base_combo.set("全部基座")
        self.base_combo.pack(side="left")
        self.base_combo.bind("<<ComboboxSelected>>", lambda e: self._browse())  # 选中即生效，不用再点搜索
        self.sort_combo = ttk.Combobox(ctl, values=SORTS, state="readonly", width=16)
        self.sort_combo.set(SORTS[0])
        self.sort_combo.pack(side="left", padx=(8, 0))
        self.sort_combo.bind("<<ComboboxSelected>>", lambda e: self._browse())

        self.q_entry = ttk.Entry(ctl)
        self.q_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.q_entry.bind("<Return>", lambda e: self._browse())

        ttk.Button(ctl, text="搜索", style="Accent.TButton", command=self._browse).pack(side="left", padx=(8, 0))

        # 分类快捷按钮：不搜索也能按分类浏览
        chips = ttk.Frame(tab, padding=(0, 0, 0, 6))
        chips.pack(fill="x")
        for label, types_val in TYPES:
            ttk.Button(chips, text=label, command=lambda t=types_val: self._browse_type(t)).pack(side="left", padx=(0, 6))

        # 翻页（不做加载更多，省资源）
        nav = ttk.Frame(tab, padding=(0, 0, 0, 6))
        nav.pack(fill="x")
        self.prev_btn = ttk.Button(nav, text="◀ 上一页", command=lambda: self._page_go(-1))
        self.prev_btn.pack(side="left")
        self.page_lbl = tk.Label(nav, text="", bg=BG, fg=MUTED, font=("", 9))
        self.page_lbl.pack(side="left", padx=10)
        self.next_btn = ttk.Button(nav, text="下一页 ▶", command=lambda: self._page_go(1))
        self.next_btn.pack(side="left")

        # 结果过滤：标签下拉（C站标签分类：风格/概念/角色/服装…）+ 可选正则（匹配 标签/标题/触发词）
        rfilter = ttk.Frame(tab, padding=(0, 0, 0, 6))
        rfilter.pack(fill="x")
        ttk.Label(rfilter, text="标签:").pack(side="left")
        self.res_tag = ttk.Combobox(rfilter, values=["全部标签"] + sorted(KIND_ZH), state="readonly", width=8)
        self.res_tag.set("全部标签")
        self.res_tag.pack(side="left", padx=(6, 0))
        self.res_tag.bind("<<ComboboxSelected>>", lambda e: self._browse())  # 标签=服务端过滤，选中即重搜
        ttk.Label(rfilter, text="正则:").pack(side="left", padx=(10, 0))
        self.res_re = ttk.Entry(rfilter, width=20)
        self.res_re.pack(side="left", padx=(6, 0))
        self.res_re.bind("<Return>", lambda e: self._apply_res_filter())
        ttk.Button(rfilter, text="过滤", command=self._apply_res_filter).pack(side="left", padx=(6, 0))
        ttk.Button(rfilter, text="清空", command=self._clear_res_filter).pack(side="left", padx=(6, 0))
        self.res_cnt = tk.Label(rfilter, text="", bg=BG, fg=MUTED, font=("", 9))
        self.res_cnt.pack(side="left", padx=(8, 0))

        self.scroll = ScrollFrame(tab)
        self.scroll.pack(fill="both", expand=True)
        self.empty_lbl = tk.Label(self.scroll.inner, text="启动后自动加载热门模型 · 也可输入关键词搜索，结果直接标 ✓已下载 / 🔄有更新",
                                  bg=BG, fg=MUTED, pady=40)
        self.empty_lbl.pack()

    def _browse(self):
        """新搜索 / 新浏览：清空分页缓存，拉第 1 页。关键词可为空（浏览热门）。"""
        if self._busy:
            return
        qtext = self.q_entry.get().strip()
        types = self._search_types
        sort = self.sort_combo.get()
        base = None if self.base_combo.get() in ("", "全部基座") else self.base_combo.get()
        tag = None if self.res_tag.get() in ("", "全部标签") else kind_canon(self.res_tag.get())
        self._res_tag = tag  # 服务端 tag 过滤（并入搜索，与基座/类别/关键词叠加）
        self._ctx = (qtext, types, sort, base, tag)
        self._pages = []
        self._page_idx = -1
        self._fetch_token += 1
        self._busy = True
        self.scroll.clear()
        self._covers.clear()
        self._cover_cache.clear()
        self.empty_lbl = tk.Label(self.scroll.inner, text="加载中…", bg=BG, fg=MUTED, pady=40)
        self.empty_lbl.pack()
        self.status_var.set("加载中…")
        self._set_nav_state()
        self._fetch(self._ctx, {t: None for t in (types or [None])})

    def _browse_type(self, types_val):
        self._search_types = types_val
        self._browse()

    def _page_go(self, delta):
        if self._busy or not self._pages:
            return
        target = self._page_idx + delta
        if target < 0:
            return
        if target < len(self._pages):  # 已有缓存，直接切页
            self._page_idx = target
            self._render_page()
            return
        cur = self._pages[self._page_idx]
        if not cur["has_more"]:
            return
        self._busy = True
        self._set_nav_state()
        self._fetch(self._ctx, dict(cur["cursors"]))

    def _fetch(self, ctx, cursors):
        qtext, types, sort, base, stag = ctx
        tag = self._fetch_token
        threading.Thread(target=self._search_thread, args=(qtext, types, sort, base, stag, cursors, tag), daemon=True).start()

    def _search_thread(self, qtext, types, sort, base, stag, cursors, tag):
        try:
            # v1 API 是 cursor 分页且不支持逗号合并 types，逐类型请求后按 id 去重合并
            keys = types if types else [None]
            cursors = dict(cursors)

            def fetch(t, cursor, limit):
                params = {"sort": sort, "limit": limit}
                if qtext:
                    params["query"] = qtext
                if t is not None:
                    params["types"] = t
                if base:
                    params["baseModels"] = base
                if stag:  # 标签过滤并入服务端搜索（与关键词/基座/类别叠加）
                    params["tag"] = stag
                if cursor:
                    params["cursor"] = cursor
                data = civitai_get("models", params)
                md = data.get("metadata") or {}
                return data.get("items", []), md.get("nextCursor")

            merged: list = []
            seen: set = set()

            def add(raw) -> bool:
                added = False
                for m in raw:
                    mid = m.get("id")
                    if mid is not None:
                        if mid in seen:
                            continue
                        seen.add(mid)
                    merged.append(m)
                    added = True
                return added

            # 首轮：每类型按均分额度拉一页
            per_type = max(1, PAGE_TARGET // max(1, len(keys)))
            for t in keys:
                raw, nc = fetch(t, cursors.get(t), per_type)
                cursors[t] = nc
                add(raw)
            # 补满规则：不足 PAGE_TARGET 时顺着剩余 cursor 补拉（最多 MAX_FILL_ROUNDS 轮）
            for _ in range(MAX_FILL_ROUNDS):
                if len(merged) >= PAGE_TARGET:
                    break
                progress = False
                for t in keys:
                    if len(merged) >= PAGE_TARGET:
                        break
                    if not cursors.get(t):
                        continue
                    raw, nc = fetch(t, cursors[t], max(1, PAGE_TARGET - len(merged)))
                    cursors[t] = nc
                    if add(raw):
                        progress = True
                if not progress:
                    break
            items = annotate_items([model_meta(m) for m in merged])
            has_more = any(cursors.values())
            self.q.put(("search", tag, items, None, cursors, has_more))
        except Exception as exc:  # noqa: BLE001
            self.q.put(("search", tag, None, str(exc), None, False))

    def _apply_res_filter(self):
        self._res_tag = None if self.res_tag.get() in ("", "全部标签") else kind_canon(self.res_tag.get())
        pat = self.res_re.get().strip()
        self._res_pat = None
        if pat:
            try:
                self._res_pat = re.compile(pat, re.IGNORECASE)
            except re.error as e:  # noqa: BLE001
                self.status_var.set(f"正则无效: {e}")
                self._res_pat = None
        if self._pages:
            self._render_page()

    def _clear_res_filter(self):
        self.res_re.delete(0, "end")
        self.res_tag.set("全部标签")
        self._res_tag = None
        self._res_pat = None
        self._browse()  # 清掉服务端 tag 过滤，重新搜索

    def _res_match(self, m) -> bool:
        # 标签已由服务端 tag 参数过滤，这里只剩客户端正则（标题/触发词/标签）
        if self._res_pat:
            hay = " ".join([str(x) for x in (m.get("name"), m.get("type"), m.get("baseModel"))]
                           + list(m.get("trainedWords") or []) + list(m.get("tags") or [])).lower()
            if not self._res_pat.search(hay):
                return False
        return True

    def _render_page(self):
        self.scroll.clear()
        self.empty_lbl = None
        self._covers.clear()  # 旧页 Label 已销毁，避免迟到的封面消息引用失效控件
        page = self._pages[self._page_idx]
        items = [m for m in page["items"] if self._res_match(m)]
        row = None
        for i, m in enumerate(items):
            if i % 2 == 0:  # 两列网格
                row = ttk.Frame(self.scroll.inner)
                row.pack(fill="x", pady=(0, 8))
            card = self._add_card(m, row)
            card.pack(side="left", fill="x", expand=True, padx=(4, 4))
            self._load_cover(m)
        shown, total = len(items), len(page["items"])
        active = bool(self._res_pat or self._res_tag)
        self.res_cnt.config(text=f"{shown}/{total}" if active else "")
        if shown == 0 and active:
            self._show_msg("过滤后本页无结果")
        fil = ""
        if active:
            parts = []
            if self._res_tag:
                parts.append(kind_display(self._res_tag))
            if self._res_pat:
                parts.append(self._res_pat.pattern)
            fil = "（已过滤: " + "/".join(parts) + "）"
        self.status_var.set(f"第 {self._page_idx + 1} 页 · 显示 {shown}/{total} 条"
                            + (" · 还有更多" if page["has_more"] else " · 到底了") + fil)
        self._set_nav_state()

    def _show_msg(self, text):
        try:
            if self.empty_lbl is None or not self.empty_lbl.winfo_exists():
                raise tk.TclError
            self.empty_lbl.config(text=text)
            self.empty_lbl.pack()
        except tk.TclError:
            self.empty_lbl = tk.Label(self.scroll.inner, text=text, bg=BG, fg=MUTED, pady=40)
            self.empty_lbl.pack()

    def _set_nav_state(self):
        if not self._pages:
            self.page_lbl.config(text="")
            self.prev_btn.state(["disabled"])
            self.next_btn.state(["disabled"])
            return
        idx = self._page_idx
        cur = self._pages[idx]
        self.page_lbl.config(text=f"第 {idx + 1} 页")
        self.prev_btn.state(["!disabled"] if idx > 0 else ["disabled"])
        self.next_btn.state(["!disabled"] if (cur["has_more"] or idx + 1 < len(self._pages)) else ["disabled"])

    def _add_card(self, m, parent=None):
        card = ttk.Frame(parent or self.scroll.inner, style="Card.TFrame", padding=8)

        # 缩略图框
        box = tk.Frame(card, width=96, height=96, bg=PANEL2)
        box.pack_propagate(False)
        box.pack(side="left")
        thumb = tk.Label(box, text="…", bg=PANEL2, fg=MUTED, font=("", 16))
        thumb.pack(fill="both", expand=True)
        mid = m.get("id")
        if mid is not None:
            self._covers[mid] = thumb

        info = ttk.Frame(card, style="Card.TFrame")
        info.pack(side="left", fill="x", expand=True, padx=(10, 0))

        name = tk.Label(info, text=m.get("name") or "(无名)", bg=PANEL, fg=FG,
                        font=("", 11, "bold"), anchor="w", wraplength=380, justify="left")
        name.pack(fill="x")

        badges = []
        if m.get("downloaded"):
            if m.get("updateAvailable"):
                badges.append(("🔄 有更新", WARN))
            else:
                badges.append(("✓ 已下载" + ("?" if m.get("localMaybe") else ""), OK))
        if (m.get("availability") or "Public") != "Public":
            badges.append(("EA 早期访问", WARN))
        if m.get("nsfw"):
            badges.append(("NSFW", ERR))
        if badges:
            row = ttk.Frame(info, style="Card.TFrame")
            row.pack(fill="x", pady=(2, 0))
            for text, color in badges:
                tk.Label(row, text=text, bg=PANEL, fg=color, font=("", 9, "bold")).pack(side="left", padx=(0, 8))

        f = m.get("file") or {}
        meta_parts = [m.get("type") or "?", f"{f.get('sizeKB', 0) / 1024:.0f}MB" if f.get("sizeKB") else "?MB"]
        if m.get("baseModel"):
            meta_parts.append(str(m["baseModel"]))
        if m.get("creator"):
            meta_parts.append("by " + str(m["creator"]))
        tk.Label(info, text=" · ".join(meta_parts), bg=PANEL, fg=MUTED, font=("", 9), anchor="w").pack(fill="x")

        if len(m.get("baseModels") or []) > 1:
            tk.Label(info, text="多基座: " + " / ".join(m["baseModels"][:4])
                     + (f" 等{len(m['baseModels'])}个" if len(m["baseModels"]) > 4 else ""),
                     bg=PANEL, fg=MUTED, font=("", 9), anchor="w").pack(fill="x")

        if m.get("updateAvailable"):
            tk.Label(info, text=f"本地 {m.get('localVersion') or '旧版'} → 最新 {m.get('version') or '新版'}",
                     bg=PANEL, fg=WARN, font=("", 9), anchor="w").pack(fill="x")

        words = (m.get("trainedWords") or [])[:3]
        if words:
            tk.Label(info, text="触发词: " + " ".join(f"`{w}`" for w in words),
                     bg=PANEL, fg=ACCENT2, font=("", 9), anchor="w", wraplength=340).pack(fill="x")

        btns = ttk.Frame(info, style="Card.TFrame")
        btns.pack(fill="x", pady=(6, 0))
        if m.get("downloaded") and not m.get("updateAvailable"):
            ttk.Button(btns, text="已下载", state="disabled").pack(side="left")
        elif not f.get("downloadUrl"):
            ttk.Button(btns, text="无文件", state="disabled").pack(side="left")
        else:
            label = "🔄 更新下载" if m.get("updateAvailable") else "⬇ 下载"
            ttk.Button(btns, text=label, style="Accent.TButton",
                       command=lambda meta=m: self._start_download(meta)).pack(side="left")
        ttk.Button(btns, text="C站详情 ↗", command=lambda u=m.get("modelUrl"): webbrowser.open(u)).pack(side="left", padx=(8, 0))
        return card

    def _load_cover(self, m):
        mid = m.get("id")
        url = m.get("cover")
        if mid is None or not url:
            return
        cached = self._cover_cache.get(mid)
        if cached is not None:
            lbl = self._covers.get(mid)
            if lbl is not None:
                lbl.config(image=cached, text="")
            return
        threading.Thread(target=self._cover_thread, args=(mid, url), daemon=True).start()

    def _cover_thread(self, mid, url):
        # 下载 + PIL 解码/缩放/转码都在工作线程；主线程只做轻量 tk.PhotoImage(data=PNG)
        # 磁盘缓存 + 原 CDN 优先、代理兜底；瞬时失败稍等重试一次
        try:
            with self._cover_sem:
                data = cover_fetch_bytes(mid, url)
            if not data:
                time.sleep(1.5)
                with self._cover_sem:
                    data = cover_fetch_bytes(mid, url)
            if not data:
                return
            png = _to_png_bytes(data)
            if png:
                self.q.put(("img_png", mid, png))
        except Exception:  # noqa: BLE001
            pass

    # ---------- 下载 ----------
    def _start_download(self, m):
        f = m.get("file") or {}
        info = self._ask_download_dir(m)  # 选 版本/基座 + 分支/用途 → 自动进 anima-style 这类目录
        if not info:
            return
        if (info.get("availability") or "Public") != "Public":
            if not messagebox.askyesno(
                    "早期访问版本",
                    "该版本是 Early Access（早期访问），通常需要 Civitai 账号订阅创作者后才能下载。\n\n仍要尝试下载吗？"):
                return
        url = info.get("url") or f.get("downloadUrl")
        if not url:
            messagebox.showwarning("无法下载", "该模型没有可下载的文件")
            return
        dest_dir = DL_DIR / info["subdir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext = ".safetensors"
        src = info.get("file") or f.get("name") or ""
        if src:
            e = Path(src).suffix.lower()
            if e in (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".sft"):
                ext = e
        fname = safe_name(m.get("name") or "model") + ext
        dest = dest_dir / fname
        if dest.exists():
            messagebox.showinfo("已存在", f"{fname} 已存在（{info['subdir']}）")
            return
        rel = dest.relative_to(DL_DIR)
        # 先落便签（类别/分支/用途/关键词），下载完 已下载页 立即带标签
        tags = load_tags()
        tags[str(rel)] = {"cat": info["cat"], "branch": info["branch"], "kind": info["kind"], "tags": info["tags"]}
        save_tags(tags)
        # 元数据先落盘，记录所选版本（多基座：更新检测按对应版本走）
        meta_save = dict(m)
        file0 = dict(f)
        file0["name"] = src
        file0["downloadUrl"] = url
        meta_save["file"] = file0
        meta_save["versionId"] = info.get("versionId") or m.get("versionId")
        meta_save["version"] = info.get("version") or m.get("version")
        meta_save["baseModel"] = info.get("baseModel") or m.get("baseModel")
        mf = meta_path_for(rel)
        mf.parent.mkdir(parents=True, exist_ok=True)
        mf.write_text(json.dumps(meta_save, ensure_ascii=False, indent=2), encoding="utf-8")
        headers = {}
        key = api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        dl_id = f"{int(time.time())}-{threading.get_ident()}"
        self._dls[dl_id] = {"file": str(dest), "total": 0, "done": 0, "status": "queued"}
        threading.Thread(target=civitai_download, args=(url, dest, dl_id, headers, self.q), daemon=True).start()
        self.status_var.set(f"开始下载: {fname} → {info['subdir']}")

    def _ask_download_dir(self, m):
        """下载前确认 版本/基座 + 类别/子目录/分支/用途/关键词：决定落盘目录（如 loras/Anima-style）并顺带打便签。
        多基座模型（同一 LoRA 适配 SDXL/Pony/Illustrious/Flux…）在版本下拉里选要下哪个。
        大模型（Checkpoint）落盘时可在 checkpoints / diffusion_models 之间选（Flux/Hunyuan 等 DiT 系默认 diffusion_models）。"""
        result: dict = {}
        cat = cat_of(m)
        versions = m.get("versions") or []
        win = tk.Toplevel(self)
        win.title("下载到哪 - " + (m.get("name") or "")[:40])
        win.configure(bg=BG)
        win.geometry(f"+{self.winfo_rootx() + 100}+{self.winfo_rooty() + 120}")
        win.grab_set()
        body = ttk.Frame(win, padding=14)
        body.pack(fill="both", expand=True)

        # 版本/基座：多基座模型选一个（决定下载哪个文件 + 分支默认值）
        ttk.Label(body, text="版本/基座").grid(row=0, column=0, sticky="w", pady=3)
        ver_cb = ttk.Combobox(body, state="readonly", width=38)
        ver_cb["values"] = [f"{x.get('name') or '?'} · {x.get('baseModel') or '?'}  ({(x.get('sizeKB') or 0) / 1024:.0f}MB)"
                            + (" [EA 早期访问]" if (x.get("availability") or "Public") != "Public" else "")
                            for x in versions] or ["(无版本)"]
        ver_cb.current(0)
        ver_cb.grid(row=0, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(body, text="类别").grid(row=1, column=0, sticky="w", pady=3)
        cat_cb = ttk.Combobox(body, values=CAT_LABELS, state="readonly", width=22)
        cat_cb.set(cat_display(cat))
        cat_cb.grid(row=1, column=1, sticky="w", padx=8, pady=3)

        # 子目录：大模型可在 checkpoints / diffusion_models 之间选（可手输其它），其它类别跟随配置
        ttk.Label(body, text="子目录").grid(row=2, column=0, sticky="w", pady=3)
        sub_cb = ttk.Combobox(body, width=22)
        sub_cb.grid(row=2, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(body, text="分支（基座：Anima/Flux/Illustrious…）").grid(row=3, column=0, sticky="w", pady=3)
        br_cb = ttk.Combobox(body, values=known_branches(), width=22)
        br_cb.set(m.get("baseModel") or "")
        br_cb.grid(row=3, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(body, text="用途（默认 风格）").grid(row=4, column=0, sticky="w", pady=3)
        kind_cb = ttk.Combobox(body, values=sorted({kind_display(k) for k in
                                                    (set(KIND_PRESETS) | {t.get("kind") for t in load_tags().values() if t.get("kind")})}),
                               width=22)
        kind_cb.set("风格")
        kind_cb.grid(row=4, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(body, text="关键词（逗号分隔，可选）").grid(row=5, column=0, sticky="w", pady=3)
        kw_en = ttk.Entry(body, width=26)
        kw_en.grid(row=5, column=1, sticky="w", padx=8, pady=3)

        preview = tk.Label(body, text="", bg=PANEL, fg=ACCENT2, font=("", 9), anchor="w")
        preview.grid(row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(10, 2))

        subdir_map = load_subdir_map()
        # DiT 系基座：ComfyUI 里这些大模型通常放 diffusion_models
        DIT_KEYS = ("Flux", "Hunyuan", "Wan", "Qwen", "SD 3", "GLM", "SeaArt", "Zeek", "DiT", "Cascade", "AuraFlow")

        def base_dir_for(cat_disp: str) -> str:
            c = cat_canon(cat_disp) or "Other"
            if c == "其他":
                c = "Other"
            return subdir_map.get(c, "other")

        def on_cat(*_):
            base = base_dir_for(cat_cb.get())
            choices = [base]
            if base_dir_for(cat_cb.get()) == "checkpoints" or cat_canon(cat_cb.get()) == "Checkpoint":
                if "diffusion_models" not in choices:
                    choices.append("diffusion_models")
            sub_cb["values"] = choices
            br = br_cb.get() or ""
            if cat_canon(cat_cb.get()) == "Checkpoint" and any(k in br for k in DIT_KEYS):
                sub_cb.set("diffusion_models")
            else:
                sub_cb.set(base)
            refresh_preview()

        def build_subdir() -> str:
            base = sub_cb.get().strip().strip("/\\") or base_dir_for(cat_cb.get())
            parts = [safe_name(p) for p in (br_cb.get().strip(), kind_canon(kind_cb.get())) if p]
            if len(parts) == 1:
                return f"{base}/{parts[0]}"
            if len(parts) >= 2:
                return f"{base}/{parts[0]}-{parts[1]}"
            return base

        def refresh_preview(*_):
            preview.config(text="将下载到: " + build_subdir())

        def on_ver(*_):
            idx = ver_cb.current()
            if 0 <= idx < len(versions):
                br_cb.set(versions[idx].get("baseModel") or br_cb.get())
            on_cat()

        ver_cb.bind("<<ComboboxSelected>>", on_ver)
        cat_cb.bind("<<ComboboxSelected>>", on_cat)
        for w in (kind_cb, sub_cb):
            w.bind("<<ComboboxSelected>>", refresh_preview)
        br_cb.bind("<<ComboboxSelected>>", refresh_preview)
        br_cb.bind("<KeyRelease>", refresh_preview)
        kind_cb.bind("<KeyRelease>", refresh_preview)
        sub_cb.bind("<KeyRelease>", refresh_preview)

        btns = ttk.Frame(win, padding=14)
        btns.pack(fill="x")

        def ok():
            idx = ver_cb.current()
            chosen = versions[idx] if 0 <= idx < len(versions) else (versions[0] if versions else {})
            result["subdir"] = build_subdir()
            result["cat"] = cat_canon(cat_cb.get())
            result["branch"] = br_cb.get().strip()
            result["kind"] = kind_canon(kind_cb.get())
            result["tags"] = [t.strip() for t in re.split(r"[,，]", kw_en.get()) if t.strip()]
            result["url"] = chosen.get("downloadUrl") or (m.get("file") or {}).get("downloadUrl")
            result["availability"] = chosen.get("availability")
            result["versionId"] = chosen.get("id")
            result["version"] = chosen.get("name")
            result["baseModel"] = chosen.get("baseModel")
            result["file"] = chosen.get("file")
            win.destroy()

        ttk.Button(btns, text="确定下载", style="Accent.TButton", command=ok).pack(side="left")
        ttk.Button(btns, text="取消", command=win.destroy).pack(side="left", padx=(8, 0))
        on_cat()
        win.wait_window()
        return result or None

    # ---------- 已下载页 ----------
    def _build_local_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  已下载  ")
        fbar = ttk.Frame(tab, padding=(0, 8, 0, 6))
        fbar.pack(fill="x")
        ttk.Label(fbar, text="筛选:").pack(side="left")
        self.f_dir = ttk.Combobox(fbar, values=["全部目录"], state="readonly", width=10)
        self.f_dir.set("全部目录")
        self.f_dir.pack(side="left", padx=(6, 0))
        self.f_dir.bind("<<ComboboxSelected>>", lambda e: self._filter_apply())
        self.f_cat = ttk.Combobox(fbar, values=["全部"] + CAT_LABELS,
                                  state="readonly", width=9)
        self.f_cat.set("全部")
        self.f_cat.pack(side="left", padx=(6, 0))
        self.f_cat.bind("<<ComboboxSelected>>", lambda e: self._filter_apply())
        self.f_branch = ttk.Combobox(fbar, values=["全部"] + known_branches(), width=12)
        self.f_branch.set("全部")
        self.f_branch.pack(side="left", padx=(6, 0))
        self.f_branch.bind("<<ComboboxSelected>>", lambda e: self._filter_apply())
        self.f_branch.bind("<Return>", lambda e: self._filter_apply())
        self.f_kind = ttk.Combobox(fbar, values=["全部"] + sorted({kind_display(k) for k in all_tag_kinds()}), width=10)
        self.f_kind.set("全部")
        self.f_kind.pack(side="left", padx=(6, 0))
        self.f_kind.bind("<<ComboboxSelected>>", lambda e: self._filter_apply())
        self.f_kw = ttk.Combobox(fbar, values=[""] + all_tag_keywords(), width=13)
        self.f_kw.set("")
        self.f_kw.pack(side="left", padx=(6, 0))
        self.f_kw.bind("<<ComboboxSelected>>", lambda e: self._filter_apply())
        self.f_kw.bind("<Return>", lambda e: self._filter_apply())
        ttk.Checkbutton(fbar, text="正则", variable=self._kw_regex, command=self._filter_apply).pack(side="left", padx=(4, 0))
        ttk.Button(fbar, text="筛选", command=self._filter_apply).pack(side="left", padx=(6, 0))
        ttk.Button(fbar, text="清空", command=self._filter_clear).pack(side="left", padx=(6, 0))
        self.filter_cnt = tk.Label(fbar, text="", bg=BG, fg=MUTED, font=("", 9))
        self.filter_cnt.pack(side="left", padx=(8, 0))
        ttk.Button(fbar, text="刷新", command=self._refresh_local).pack(side="right")

        nav = ttk.Frame(tab, padding=(0, 0, 0, 6))
        nav.pack(fill="x")
        self.local_prev = ttk.Button(nav, text="◀ 上一页", command=lambda: self._local_go(-1))
        self.local_prev.pack(side="left")
        self.local_page_lbl = tk.Label(nav, text="", bg=BG, fg=MUTED, font=("", 9))
        self.local_page_lbl.pack(side="left", padx=10)
        self.local_next = ttk.Button(nav, text="下一页 ▶", command=lambda: self._local_go(1))
        self.local_next.pack(side="left")
        ttk.Checkbutton(nav, text="显示封面（大库存建议关）", variable=self.show_cover,
                        command=self._refresh_local).pack(side="left", padx=(16, 0))

        self.local_scroll = ScrollFrame(tab)
        self.local_scroll.pack(fill="both", expand=True)

    def _filter_apply(self):
        self._local_page = 0  # 筛选变化回到第一页
        self._lfilter = {
            "dir": "" if self.f_dir.get() in ("", "全部目录") else self.f_dir.get(),
            "cat": "" if self.f_cat.get() in ("", "全部") else cat_canon(self.f_cat.get()),
            "branch": "" if self.f_branch.get() in ("", "全部") else self.f_branch.get(),
            "kind": "" if self.f_kind.get() in ("", "全部") else kind_canon(self.f_kind.get()),
            "kw": "" if self.f_kw.get() in ("", "全部") else self.f_kw.get().strip(),
        }
        self._refresh_local()

    def _filter_clear(self):
        self._local_page = 0  # 筛选变化回到第一页
        self.f_dir.set("全部目录")
        self.f_cat.set("全部")
        self.f_branch.set("全部")
        self.f_kind.set("全部")
        self.f_kw.set("")
        self._lfilter = {"cat": "", "branch": "", "kind": "", "kw": "", "dir": ""}
        self._refresh_local()

    def _match_filter(self, rel_str, meta, rec, f) -> bool:
        """五维叠加（AND）：目录 / 类别 / 分支 / 用途 / 关键词(可正则)。"""
        parts_p = Path(rel_str).parts
        top = parts_p[0] if parts_p else ""
        if f.get("dir") and top != f["dir"]:
            return False
        if f["cat"] and rec.get("cat") != f["cat"]:
            return False
        if f["branch"] and f["branch"].lower() not in (rec.get("branch") or "").lower():
            return False
        if f["kind"] and rec.get("kind") != f["kind"]:
            return False
        if f["kw"]:
            # 关键词/正则：匹配 标签 + 名称 + 触发词 + 目录路径（路径统一正斜杠，loras、loras/Anima、checkpoints…）
            parent = str(Path(rel_str).parent).replace("\\", "/")
            # 路径放最前：^ 锚点对目录永远有效（loras、loras/Anima、checkpoints…）
            hay = " ".join([parent, top, Path(rel_str).name]
                           + [str(x) for x in ((rec.get("branch") or ""), (rec.get("kind") or ""),
                                               kind_display(rec.get("kind")))]
                           + list(rec.get("tags") or []) + [str((meta or {}).get("name") or "")]).strip().lower()
            if self._kw_regex.get():
                try:
                    if not re.search(f["kw"].lower(), hay):
                        return False
                except re.error:
                    return False
            else:
                toks = [t for t in re.split(r"[,，\s]+", f["kw"]) if t]
                if toks and not all(t.lower() in hay for t in toks):
                    return False
        return True

    def _tags_summary(self, rec) -> str:
        parts = []
        if rec.get("cat"):
            parts.append(cat_display(rec["cat"]))
        if rec.get("branch"):
            parts.append(rec["branch"])
        if rec.get("kind"):
            parts.append(kind_display(rec["kind"]))
        if rec.get("tags"):
            parts.append(" ".join(rec["tags"]))
        return "标签: " + " · ".join(parts) if parts else "标签: 未标"

    def _refresh_local(self):
        self.local_scroll.clear()
        self._local_covers.clear()
        inner = self.local_scroll.inner
        tags = load_tags()
        f = self._lfilter
        # 筛选下拉随时收编标签里用过的 用途/关键词/分支（中文显示，不用手输）
        self.f_kind.configure(values=["全部"] + sorted({kind_display(k) for k in all_tag_kinds()}))
        self.f_kw.configure(values=[""] + all_tag_keywords())
        self.f_branch.configure(values=["全部"] + known_branches())
        try:
            dirs = sorted(d.name for d in DL_DIR.iterdir() if d.is_dir() and d.name != "_meta")
        except OSError:
            dirs = []
        self.f_dir.configure(values=["全部目录"] + dirs)
        rows = []  # (sub, rel, size, meta, rec)
        for file in sorted(DL_DIR.rglob("*.safetensors")) + sorted(DL_DIR.rglob("*.ckpt")):
            rel = file.relative_to(DL_DIR)
            meta = load_meta(rel) or load_sidecar_meta(file)
            if not meta:
                meta = {}
            # 老 Lora 兼容：无 sidecar/meta 时从 safetensors 头部补 名称/基座/触发词
            if not (meta.get("name") or meta.get("trainedWords")):
                enr = sf_enrich_cached(file)
                if enr:
                    meta = {**meta, **enr}
            # 预览图：相邻本地图优先（零流量），其次 meta 里的封面
            pv = find_preview(file)
            if pv:
                meta["cover"] = str(pv)
            rec = tags.get(str(rel)) or {}
            if not self._match_filter(str(rel), meta, rec, f):
                continue
            sub = str(rel.parent) if str(rel.parent) != "." else "根目录"
            rows.append((sub, rel, file.stat().st_size, meta, rec))
        total = len(rows)
        if not total:
            tk.Label(inner, text="没有匹配的模型" + ("（换个筛选或点清空）" if any(f.values()) else "。去搜索页拉几个吧"),
                     bg=BG, fg=MUTED, pady=40).pack()
            self.filter_cnt.config(text="")
            self._local_nav_state(0)
            return
        self.filter_cnt.config(text=f"{total} 条")
        # 分页（大库存省资源）：每页 100
        per = 100
        pages = max(1, (total + per - 1) // per)
        if self._local_page >= pages:
            self._local_page = pages - 1
        start = self._local_page * per
        page_rows = rows[start:start + per]
        self._local_nav_state(total)
        show_cover = self.show_cover.get()
        for sub in sorted({r[0] for r in page_rows}):
            tk.Label(inner, text=f"📁 {sub}", bg=BG, fg=ACCENT2, font=("", 10, "bold"), anchor="w").pack(fill="x", pady=(8, 2))
            for _sub, rel, size, meta, rec in [r for r in page_rows if r[0] == sub]:
                row = ttk.Frame(inner, style="Card.TFrame", padding=4)
                row.pack(fill="x", pady=2)
                if show_cover:
                    box = tk.Frame(row, width=44, height=44, bg=PANEL2)
                    box.pack_propagate(False)
                    box.pack(side="left")
                    thumb = tk.Label(box, text="", bg=PANEL2, fg=MUTED, font=("", 8))
                    thumb.pack(fill="both", expand=True)
                    rel_str = str(rel)
                    cached = self._local_cover_cache.get(rel_str)
                    if cached is not None:
                        thumb.config(image=cached)
                    else:
                        self._local_covers[rel_str] = thumb
                        self._load_local_cover(rel_str, meta)
                info = f"{rel.name}   ·   {size / 1048576:.0f} MB"
                if meta and meta.get("name"):
                    info += f"   ·   {meta.get('name')}"
                    if meta.get("version"):
                        info += f"   [{meta.get('version')}]"
                tw = (meta or {}).get("trainedWords") or []
                if tw:
                    info += "   触发词: " + " ".join(str(x) for x in tw[:3])
                tk.Label(row, text=info, bg=PANEL, fg=FG, font=("", 9), anchor="w").pack(side="left", fill="x", expand=True)
                tk.Label(row, text=self._tags_summary(rec), bg=PANEL, fg=ACCENT2, font=("", 9)).pack(side="left", padx=(8, 0))
                ttk.Button(row, text="标签", width=5,
                           command=lambda r0=str(rel), m=meta: self._edit_tags(r0, m)).pack(side="left", padx=(6, 0))

    def _local_go(self, delta):
        self._local_page = max(0, self._local_page + delta)
        self._refresh_local()

    def _local_nav_state(self, total):
        per = 100
        if not total:
            self.local_page_lbl.config(text="")
            self.local_prev.state(["disabled"])
            self.local_next.state(["disabled"])
            return
        pages = max(1, (total + per - 1) // per)
        self.local_page_lbl.config(text=f"第 {self._local_page + 1}/{pages} 页")
        self.local_prev.state(["!disabled"] if self._local_page > 0 else ["disabled"])
        self.local_next.state(["!disabled"] if self._local_page + 1 < pages else ["disabled"])

    def _load_local_cover(self, rel_str, meta):
        url = (meta or {}).get("cover")
        if not url:
            return
        threading.Thread(target=self._local_cover_thread, args=(rel_str, url), daemon=True).start()

    def _local_cover_thread(self, rel_str, url):
        # 下载 + 缩放转码都在工作线程，主线程只建轻量 PhotoImage
        # cover 可能是网络 URL，也可能是 sidecar 里存的本地预览图路径（ComfyUI-Manager 格式）
        try:
            if url.startswith(("http://", "https://")):
                with self._cover_sem:
                    data = cover_fetch_bytes(rel_str, url)
            else:
                p = Path(url)
                if not p.exists():
                    return
                data = p.read_bytes()  # 本地预览图：零流量
            if not data:
                return
            png = _to_png_bytes(data)
            if png:
                self.q.put(("local_img", rel_str, png))
        except Exception:  # noqa: BLE001
            pass

    def _edit_tags(self, rel_str, meta):
        """便签编辑：类别 / 分支 / 用途 / 关键词（可叠加）。"""
        tags = load_tags()
        rec = dict(tags.get(rel_str) or {})
        win = tk.Toplevel(self)
        win.title("标签 - " + Path(rel_str).name)
        win.configure(bg=BG)
        win.geometry(f"+{self.winfo_rootx() + 80}+{self.winfo_rooty() + 100}")
        win.grab_set()
        body = ttk.Frame(win, padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="类别").grid(row=0, column=0, sticky="w", pady=3)
        cat = ttk.Combobox(body, values=CAT_LABELS, state="readonly", width=20)
        cat.set(cat_display(rec.get("cat") or cat_of(meta)))
        cat.grid(row=0, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(body, text="分支（基座：Anima/Flux/Illustrious…）").grid(row=1, column=0, sticky="w", pady=3)
        br = ttk.Combobox(body, values=known_branches(), width=20)
        br.set(rec.get("branch") or branch_hint(rel_str, meta))
        br.grid(row=1, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(body, text="用途（风格/角色/衣服…）").grid(row=2, column=0, sticky="w", pady=3)
        kinds = sorted({kind_display(k) for k in
                        (set(KIND_PRESETS) | {t.get("kind") for t in load_tags().values() if t.get("kind")})})
        kd = ttk.Combobox(body, values=kinds, width=20)
        kd.set(kind_display(rec.get("kind")) or "")
        kd.grid(row=2, column=1, sticky="w", padx=8, pady=3)

        ttk.Label(body, text="关键词（逗号分隔，可多个）").grid(row=3, column=0, sticky="w", pady=3)
        kw = ttk.Entry(body, width=26)
        kw.insert(0, ", ".join(rec.get("tags") or []))
        kw.grid(row=3, column=1, sticky="w", padx=8, pady=3)

        btns = ttk.Frame(win, padding=14)
        btns.pack(fill="x")

        def do_save():
            new_rec = {"cat": cat_canon(cat.get()),
                       "branch": br.get().strip(),
                       "kind": kind_canon(kd.get()),
                       "tags": [t.strip() for t in re.split(r"[,，]", kw.get()) if t.strip()]}
            if any(new_rec.values()):
                tags[rel_str] = new_rec
            else:
                tags.pop(rel_str, None)
            save_tags(tags)
            win.destroy()
            self._local_dirty = False
            self._refresh_local()

        def do_clear():
            tags.pop(rel_str, None)
            save_tags(tags)
            win.destroy()
            self._local_dirty = False
            self._refresh_local()

        ttk.Button(btns, text="保存", style="Accent.TButton", command=do_save).pack(side="left")
        ttk.Button(btns, text="清除标签", command=do_clear).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="取消", command=win.destroy).pack(side="left", padx=(8, 0))

    # ---------- 设置页 ----------
    def _build_settings_tab(self):
        tab = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(tab, text="  设置  ")
        box = ttk.Frame(tab, style="Card.TFrame", padding=14)
        box.pack(fill="x")

        ttk.Label(box, text="Civitai API Key（可选，部分模型需登录/NSFW 才可下载）",
                  style="Card.TLabel").pack(anchor="w")
        ttk.Label(box, text="在 https://civitai.com/settings 生成。只保存在本机 下载目录/api_key.txt，不上传。",
                  style="Card.TLabel", foreground=MUTED).pack(anchor="w", pady=(2, 6))
        keyrow = ttk.Frame(box, style="Card.TFrame")
        keyrow.pack(fill="x")
        self.key_entry = ttk.Entry(keyrow, show="*", width=48)
        self.key_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(keyrow, text="保存", style="Accent.TButton", command=self._save_key).pack(side="left", padx=(8, 0))
        ttk.Button(keyrow, text="清除", command=self._clear_key).pack(side="left", padx=(8, 0))

        ttk.Separator(box).pack(fill="x", pady=12)
        ttk.Label(box, text="API 源", style="Card.TLabel").pack(anchor="w")
        ttk.Label(box, text="civitai.com 需梯子；civitai.red 是官方国内镜像，直连搜索/下载不烧梯子流量。",
                  style="Card.TLabel", foreground=MUTED).pack(anchor="w", pady=(2, 6))
        apir = ttk.Frame(box, style="Card.TFrame")
        apir.pack(fill="x")
        self.api_combo = ttk.Combobox(apir, values=[a[0] for a in API_SOURCES], state="readonly", width=26)
        self.api_combo.set(next((a[0] for a in API_SOURCES if a[1] == CIVITAI_API), API_SOURCES[0][0]))
        self.api_combo.pack(side="left")
        ttk.Button(apir, text="切换", command=self._apply_api).pack(side="left", padx=(8, 0))

        ttk.Separator(box).pack(fill="x", pady=12)
        self.dir_label = ttk.Label(box, text=f"下载/扫描目录：{DL_DIR}", style="Card.TLabel")
        self.dir_label.pack(anchor="w")
        ttk.Label(box, text="扫描并下载到这个目录（默认你的 ComfyUI models 根）：LoRA 进 loras/，大模型进 checkpoints/，可直接给 ComfyUI 用。",
                  style="Card.TLabel", foreground=MUTED).pack(anchor="w", pady=(2, 6))
        dirrow = ttk.Frame(box, style="Card.TFrame")
        dirrow.pack(fill="x")
        ttk.Button(dirrow, text="更换目录", command=self._pick_dir).pack(side="left")
        ttk.Button(dirrow, text="打开目录", command=self._open_dir).pack(side="left", padx=(8, 0))

        ttk.Separator(box).pack(fill="x", pady=12)
        ttk.Label(box, text="下载目录映射（各类模型落盘的子目录，相对下载目录）",
                  style="Card.TLabel").pack(anchor="w")
        ttk.Label(box, text="默认 LoRA→loras / 大模型→checkpoints / VAE→vae / Embedding→embeddings / ControlNet→controlnet。"
                           "改成你想要的子目录名后点保存，之后下载按新路径落盘（不影响已下载文件）。",
                  style="Card.TLabel", foreground=MUTED).pack(anchor="w", pady=(2, 6))
        self.subdir_entries: dict[str, ttk.Entry] = {}
        _subdir_map = load_subdir_map()
        for _label, _cat in (("LoRA", "LoRA"), ("大模型", "Checkpoint"), ("VAE", "VAE"),
                             ("Embedding", "Embedding"), ("ControlNet", "ControlNet"), ("其他", "Other")):
            srow = ttk.Frame(box, style="Card.TFrame")
            srow.pack(fill="x", pady=1)
            ttk.Label(srow, text=_label, width=12, style="Card.TLabel").pack(side="left")
            sent = ttk.Entry(srow, width=34)
            sent.insert(0, _subdir_map.get(_cat, ""))
            sent.pack(side="left", padx=(6, 0))
            self.subdir_entries[_cat] = sent
        ttk.Button(box, text="保存目录映射", style="Accent.TButton", command=self._save_subdirs).pack(anchor="w", pady=(6, 0))

        ttk.Separator(box).pack(fill="x", pady=12)
        ttk.Label(box, text="图片代理（预览图加载不出时的兜底通道）", style="Card.TLabel").pack(anchor="w")
        ttk.Label(box, text="默认自动用 https://images.weserv.nl 免费代理兜底（原 CDN 不可达时自动切换）；"
                           "可填自己的代理模板（{url} 是原图地址占位），填 off 关闭代理只用原 CDN。",
                  style="Card.TLabel", foreground=MUTED).pack(anchor="w", pady=(2, 6))
        prows = ttk.Frame(box, style="Card.TFrame")
        prows.pack(fill="x")
        self.img_proxy_entry = ttk.Entry(prows, width=52)
        self.img_proxy_entry.insert(0, load_img_proxy())
        self.img_proxy_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(prows, text="保存", style="Accent.TButton", command=self._save_img_proxy).pack(side="left", padx=(8, 0))

        crows = ttk.Frame(box, style="Card.TFrame")
        crows.pack(fill="x", pady=(6, 0))
        ttk.Label(crows, text="封面缓存上限(MB)", width=16, style="Card.TLabel").pack(side="left")
        self.cache_mb_entry = ttk.Entry(crows, width=10)
        self.cache_mb_entry.insert(0, str(max(10, cover_cache_limit_bytes() // 1048576)))
        self.cache_mb_entry.pack(side="left", padx=(6, 0))
        ttk.Label(crows, text="超限自动清最旧（默认 200）", style="Card.TLabel", foreground=MUTED).pack(side="left", padx=(8, 0))
        ttk.Button(crows, text="保存", style="Accent.TButton", command=self._save_cache_mb).pack(side="left", padx=(8, 0))

    def _save_key(self):
        save_api_key(self.key_entry.get().strip())
        self._refresh_top_tags()
        messagebox.showinfo("已保存", "API key 已保存到本机")

    def _clear_key(self):
        save_api_key("")
        self.key_entry.delete(0, "end")
        self._refresh_top_tags()

    def _apply_api(self):
        global CIVITAI_API  # noqa: PLW0603
        for label, base in API_SOURCES:
            if label == self.api_combo.get():
                CIVITAI_API = base
                save_config_api(base)
                self._cover_cache.clear()
                self.status_var.set(f"API 源已切换: {label}")
                messagebox.showinfo("已切换", f"API 源: {label}\n重新搜索/浏览即生效，下载地址自动跟随")
                break

    def _open_dir(self):
        try:
            os.startfile(str(DL_DIR))  # Windows
        except AttributeError:
            webbrowser.open(f"file://{DL_DIR}")  # macOS / Linux
        except Exception:  # noqa: BLE001
            messagebox.showerror("错误", "无法打开目录")

    def _pick_dir(self):
        p = filedialog.askdirectory(initialdir=str(DL_DIR),
                                    title="选择下载/扫描目录（建议选 ComfyUI 的 models 根）")
        if not p:
            return
        set_dl_dir(Path(p))
        self.dir_label.config(text=f"下载/扫描目录：{DL_DIR}")
        self._refresh_top_tags()
        self._local_dirty = True
        messagebox.showinfo("已切换", f"目录已切换为 {DL_DIR}\n重新搜索即可看到该目录里已下载的模型")

    def _save_subdirs(self):
        m = {}
        for cat, ent in self.subdir_entries.items():
            m[cat] = ent.get().strip().strip("/\\")
        save_subdir_map(m)
        self.status_var.set("下载目录映射已保存（新下载按新路径落盘）")
        messagebox.showinfo("已保存", "下载目录映射已保存\n后续下载将按新路径落盘（不影响已下载文件）")

    def _save_img_proxy(self):
        save_img_proxy(self.img_proxy_entry.get().strip())
        self._cover_cache.clear()  # 清了内存缓存，下次翻页按新代理重新抓
        self.status_var.set("图片代理已保存")
        messagebox.showinfo("已保存", "图片代理已保存\n下次加载封面生效（已缓存的封面不受影响）")

    def _save_cache_mb(self):
        v = self.cache_mb_entry.get().strip()
        try:
            mb = max(10, int(float(v)))
        except (TypeError, ValueError):
            messagebox.showwarning("无效", "请输入数字（MB）")
            return
        _config_set(CACHE_MB_KEY, str(mb))
        self.status_var.set(f"封面缓存上限已设为 {mb}MB")
        messagebox.showinfo("已保存", f"封面缓存上限 {mb}MB\n超限自动按最旧优先滚动清除")

    # ---------- 事件 ----------
    def _on_tab(self, _e):
        if self.notebook.index(self.notebook.select()) == 1 and self._local_dirty:
            self._refresh_local()
            self._local_dirty = False

    def _refresh_top_tags(self):
        self.dir_tag.config(text=str(DL_DIR))
        self.key_tag.config(text="🔑 key 已配置" if api_key() else "🔑 key 未配置")

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "search":
                    _, tag, items, err, cursors, has_more = msg
                    if tag != self._fetch_token:
                        continue  # 过期响应，丢弃
                    self._busy = False
                    if err:
                        self.status_var.set(f"加载失败: {err}")
                        self._show_msg(f"加载失败: {err}")
                        self._set_nav_state()
                        continue
                    if not items:
                        self.status_var.set("没有结果" + ("（清空关键词可浏览热门）" if self._ctx and self._ctx[0] else ""))
                        self._show_msg("没有结果")
                        self._set_nav_state()
                        continue
                    self._pages.append({"items": items, "cursors": cursors or {}, "has_more": has_more})
                    self._page_idx = len(self._pages) - 1
                    self._render_page()
                elif kind == "img_png":
                    _, mid, png = msg
                    try:
                        photo = tk.PhotoImage(data=png)  # 小 PNG，主线程创建很快
                    except tk.TclError:
                        continue
                    self._cover_cache[mid] = photo
                    if len(self._cover_cache) > 300:  # 简单防膨胀
                        for k in list(self._cover_cache)[:100]:
                            self._cover_cache.pop(k, None)
                    lbl = self._covers.get(mid)
                    if lbl is not None:
                        try:
                            lbl.config(image=photo, text="")
                        except tk.TclError:
                            pass  # 控件已被翻页销毁，忽略
                elif kind == "local_img":
                    _, rel_str, png = msg
                    try:
                        photo = tk.PhotoImage(data=png)  # 已下载页小图
                    except tk.TclError:
                        continue
                    self._local_cover_cache[rel_str] = photo
                    if len(self._local_cover_cache) > 400:  # 防膨胀
                        for k in list(self._local_cover_cache)[:100]:
                            self._local_cover_cache.pop(k, None)
                    lbl = self._local_covers.get(rel_str)
                    if lbl is not None:
                        try:
                            lbl.config(image=photo)
                        except tk.TclError:
                            pass
                elif kind == "dl":
                    _, dl_id, st = msg
                    self._dls[dl_id] = st
                    active = [d for d in self._dls.values() if d["status"] == "downloading"]
                    if active:
                        d = active[-1]
                        pct = round(d["done"] / d["total"] * 100) if d["total"] else 0
                        self.prog.config(value=pct)
                        self.status_var.set(f"{pct}%  {Path(d['file']).name}  ({d['done'] // 1048576}/{d['total'] // 1048576}MB)")
                    elif st["status"] == "done":
                        self.prog.config(value=100)
                        self.status_var.set(f"✅ 完成: {Path(st['file']).name}（重新搜索可见 ✓已下载 标记）")
                        self._local_dirty = True
                    elif st["status"].startswith("error"):
                        self.prog.config(value=0)
                        self.status_var.set(f"⚠️ 下载失败: {Path(st['file']).name} — {st['status']}")
        except queue.Empty:
            pass
        self.after(200, self._poll)


if __name__ == "__main__":
    CoralApp().mainloop()
