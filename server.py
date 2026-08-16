#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coral LoRA 拉取器（作者：@Ne）—— 轻量可视化 Civitai LoRA 下载器。

零依赖（仅 Python 标准库）：http.server + urllib，Windows 双击启动，
浏览器访问 http://localhost:8788 即可搜索 / 下载 / 管理 Civitai LoRA。

功能（就 4 个，够用就好）：
  1. 搜索：关键词 → C站 LoRA 列表（封面 + 名字 + 触发词）
  2. 下载：点一下 → .safetensors 本体（带进度）+ 元数据 JSON
  3. 元数据：名字 / 触发词 / 描述 / 示例图 URL / 作者 / 版本
  4. 已下载：本地列表，可重新查看元数据

API key：可选。有些模型（尤其 NSFW / 登录要求）需要 API key 才能下载。
在设置里填上 C站 API key（https://civitai.com/settings 生成）即可解锁。

用法:
    python server.py            # 启动，浏览器开 http://localhost:8788
    环境变量:
      CORAL_LORA_PORT  端口（默认 8788）
      CORAL_LORA_DIR   下载目录（默认 ~/Downloads/CoralLoRA）
      CORAL_LORA_KEY   Civitai API key（等价于网页里填）
"""
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------- 配置 ----------
PORT = int(os.environ.get("CORAL_LORA_PORT", "8788"))
HOME = Path.home()
DL_DIR = Path(os.environ.get("CORAL_LORA_DIR", str(HOME / "Downloads" / "CoralLoRA")))
CIVITAI_API = os.environ.get("CORAL_LORA_API", "https://civitai.com/api/v1").rstrip("/")
UA = "Mozilla/5.0 (CoralLoRA/0.1; +https://github.com/)"  # 礼貌 UA
TIMEOUT = 30
META_DIR = DL_DIR / "_meta"  # 元数据 JSON 集中放一个隐藏目录

DL_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# 进程内下载状态（id -> {file, total, done, status}），供前端轮询进度
DOWNLOADS: dict[str, dict] = {}
DL_LOCK = threading.Lock()


# ---------- 工具 ----------
def api_key() -> str:
    """优先级：环境变量 > 下载目录 key 文件（网页设置里写入）> 项目目录 key 文件（随包携带）。"""
    env = os.environ.get("CORAL_LORA_KEY", "").strip()
    if env:
        return env
    keyfile = DL_DIR / "api_key.txt"
    if keyfile.exists():
        return keyfile.read_text(encoding="utf-8").strip()
    # 随包：项目目录下的 api_key.txt（拷走整个文件夹 key 跟着走）
    bundled = Path(__file__).resolve().parent / "api_key.txt"
    if bundled.exists():
        return bundled.read_text(encoding="utf-8").strip()
    return ""


def save_api_key(key: str) -> None:
    keyfile = DL_DIR / "api_key.txt"
    if key:
        keyfile.write_text(key.strip(), encoding="utf-8")
    elif keyfile.exists():
        keyfile.unlink()


def civitai_get(path: str, params: dict | None = None, headers: dict | None = None) -> dict:
    """GET Civitai API，返回 JSON；异常抛 ValueError。"""
    url = f"{CIVITAI_API}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json",
        **(headers or {}),
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def civitai_download(url: str, dest: Path, dl_id: str, headers: dict | None = None) -> None:
    """流式下载到 dest，实时更新 DOWNLOADS 进度。"""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        **(headers or {}),
    })
    with DL_LOCK:
        DOWNLOADS[dl_id] = {"file": str(dest), "total": 0, "done": 0, "status": "downloading"}
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            with DL_LOCK:
                DOWNLOADS[dl_id]["total"] = total
            tmp = dest.with_suffix(dest.suffix + ".part")
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    with DL_LOCK:
                        DOWNLOADS[dl_id]["done"] = done
            tmp.replace(dest)  # 原子落盘
            with DL_LOCK:
                DOWNLOADS[dl_id]["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        with DL_LOCK:
            DOWNLOADS[dl_id]["status"] = f"error: {exc}"
        raise


def safe_name(name: str) -> str:
    """文件名净化：去掉 Windows 非法字符。"""
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return name[:120] or "model"


def model_meta(model: dict) -> dict:
    """从 API 模型对象提取我们关心的元数据（轻量：只要基础字段）。"""
    v = (model.get("modelVersions") or [{}])[0]
    files = v.get("files") or []
    main_file = None
    for f in files:
        if f.get("type") == "Model" or f.get("name", "").endswith((".safetensors", ".ckpt")):
            main_file = f
            break
    if main_file is None and files:
        main_file = files[0]
    # 触发词：从训练元数据里拿（traits / trainedWords）
    trained = v.get("trainedWords") or []
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "type": model.get("type"),
        "nsfw": model.get("nsfw"),
        "creator": (model.get("creator") or {}).get("username"),
        "stats": model.get("stats") or {},
        "description": (model.get("description") or "")[:2000],
        "cover": (model.get("images") or [{}])[0].get("url") if model.get("images") else None,
        "version": v.get("name"),
        "baseModel": v.get("baseModel"),
        "trainedWords": trained,
        "file": {
            "name": main_file.get("name") if main_file else None,
            "sizeKB": main_file.get("sizeKB") if main_file else None,
            "downloadUrl": main_file.get("downloadUrl") if main_file else None,
        } if main_file else None,
        "modelUrl": f"https://civitai.com/models/{model.get('id')}",
    }


def local_downloads() -> list[dict]:
    """扫描本地已下载：DL_DIR 里的 .safetensors/.ckpt + _meta 里的元数据。"""
    items = []
    for f in sorted(DL_DIR.glob("*.safetensors")) + sorted(DL_DIR.glob("*.ckpt")):
        meta = None
        mf = META_DIR / (f.stem + ".json")
        if mf.exists():
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = None
        items.append({
            "file": f.name,
            "sizeMB": round(f.stat().st_size / 1024 / 1024, 1),
            "meta": meta,
        })
    return items


# ---------- HTTP 处理 ----------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 安静一点
        pass

    def _send(self, code: int, payload: dict | str, ctype: str = "application/json"):
        body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, name: str):
        root = Path(__file__).resolve().parent
        p = root / name
        if not p.exists():
            self._send(404, {"ok": False, "error": f"missing {name}"})
            return
        data = p.read_bytes()
        ctype = {
            ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
            ".png": "image/png", ".svg": "image/svg+xml",
        }.get(p.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8" if "text" in ctype else ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._static("index.html")
            return

        # API: 搜索
        if path == "/api/search":
            try:
                query = qs.get("q", [""])[0].strip()
                sort = qs.get("sort", ["Most Downloaded"])[0]
                if not query:
                    self._send(400, {"ok": False, "error": "缺少关键词"})
                    return
                data = civitai_get("models", {
                    "query": query, "types": "LORA", "sort": sort, "limit": 24,
                })
                items = [model_meta(m) for m in data.get("items", [])]
                self._send(200, {"ok": True, "items": items})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": f"搜索失败: {exc}"})
            return

        # API: 已下载列表
        if path == "/api/local":
            self._send(200, {"ok": True, "items": local_downloads()})
            return

        # API: 下载进度
        if path == "/api/progress":
            with DL_LOCK:
                snap = {k: dict(v) for k, v in DOWNLOADS.items()}
            self._send(200, {"ok": True, "downloads": snap})
            return

        # API: 当前设置（有无 key / 目录）
        if path == "/api/settings":
            self._send(200, {"ok": True, "hasKey": bool(api_key()), "dir": str(DL_DIR)})
            return

        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(body or "{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "请求体不是合法 JSON"})
            return

        # 下载：{url, name, meta}
        if path == "/api/download":
            url = (data.get("url") or "").strip()
            name = (data.get("name") or "").strip()
            meta = data.get("meta") or {}
            if not url:
                self._send(400, {"ok": False, "error": "缺少下载地址"})
                return
            fname = safe_name(name or "model") + ".safetensors"
            if not fname.lower().endswith((".safetensors", ".ckpt")):
                fname += ".safetensors"
            dest = DL_DIR / fname
            if dest.exists():
                self._send(409, {"ok": False, "error": f"已存在: {fname}"})
                return
            dl_id = f"{int(time.time())}-{os.getpid()}-{len(DOWNLOADS)}"
            # 保存元数据（先落盘，下载后台跑）
            if meta:
                mf = META_DIR / (dest.stem + ".json")
                mf.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            headers = {}
            key = api_key()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            t = threading.Thread(target=civitai_download, args=(url, dest, dl_id, headers), daemon=True)
            t.start()
            self._send(200, {"ok": True, "id": dl_id, "file": dest.name})
            return

        # 保存 API key：{key}
        if path == "/api/key":
            save_api_key((data.get("key") or "").strip())
            self._send(200, {"ok": True, "hasKey": bool(api_key())})
            return

        self._send(404, {"ok": False, "error": "not found"})


def main() -> None:
    print(f"🪸 Coral LoRA 拉取器")
    print(f"   下载目录: {DL_DIR}")
    print(f"   API key : {'已配置' if api_key() else '未配置（部分模型需登录下载）'}")
    print(f"   浏览器打开: http://localhost:{PORT}")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
