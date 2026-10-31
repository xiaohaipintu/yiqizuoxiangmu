"""拼图还原 · 局域网网页版服务器。

启动：
    python web/server.py            # 默认 8000 端口
    python web/server.py --port 8080

启动后终端会打印局域网地址和二维码，手机扫码即可打开：
    上传照片 → 服务器跑流水线 → 返回拼好的图
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sys

WEB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_DIR))

import pipeline  # noqa: E402  (同目录的流水线封装)

INDEX_FILE = WEB_DIR / "index.html"
MAX_UPLOAD = 25 * 1024 * 1024          # 单张照片上限 25MB
JOB_TTL = 24 * 3600                    # 任务结果（图片）保留 24 小时
ALLOWED_FILES = {
    "upload.jpg", "assembled.jpg", "41_union_rank.png",
    "upload_labels.jpg", "assembled_labels.jpg",
    "01_mask.png", "02_pieces.png", "beam_solutions.json",
    "best_union.json", "pieces.json", "pieces_scaled.json", "result.json",
    "chosen_solution.json",
    "run.log", "segment.log", "rescale_pieces.log", "beam_search.log",
    "eval_union.log", "error.log",
}


# --------------------------------------------------------------------------
# 日志：任务目录里的 logs/ 会额外归档一份到这里，任务被清理后仍能翻
# --------------------------------------------------------------------------
def log_dir() -> Path:
    p = pipeline.job_root() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_event(msg: str) -> None:
    """服务器运行日志（收到任务/完成/出错），按天一个文件。"""
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_dir() / f"server-{time.strftime('%Y%m%d')}.log", "a",
                  encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {msg}\n")
    except Exception:
        pass


def archive_job_log(job: dict) -> None:
    """把任务目录里的 logs/ 复制一份到永久归档目录。"""
    import shutil
    try:
        src = Path(job["dir"]) / "logs"
        if not src.exists():
            return
        dst = log_dir() / f"job-{job['id']}"
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception:
        pass

JOBS: "OrderedDict[str, dict]" = OrderedDict()
JOBS_LOCK = threading.Lock()
WORKER_LOCK = threading.Lock()          # 流水线串行跑，避免抢 CPU
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()


# --------------------------------------------------------------------------
# 局域网地址
# --------------------------------------------------------------------------
def lan_ip() -> str:
    """取本机在局域网里的 IP（不真的发包，只是让系统选出口）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------
def new_job() -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",        # queued | running | done | error
        "step": "排队中…",
        "progress": 0,
        "log": [],
        "created": time.time(),
        "result": None,
        "error": None,
        "dir": str(pipeline.job_root() / job_id),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        # 清理过期任务
        now = time.time()
        for key in [k for k, v in JOBS.items() if now - v["created"] > JOB_TTL]:
            stale = JOBS.pop(key)
            _cleanup(stale)
    return job


def _cleanup(job: dict) -> None:
    import shutil
    shutil.rmtree(job.get("dir", ""), ignore_errors=True)


def worker() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            continue
        WORKER_LOCK.acquire()
        try:
            job["status"] = "running"
            out_dir = Path(job["dir"])
            out_dir.mkdir(parents=True, exist_ok=True)

            def log(name: str, extra: str = ""):
                job["step"] = f"{name} {extra}".strip()
                job["progress"] = min(95, job["progress"] + 15)
                job["log"].append(f"[{name}] {extra}")

            result = pipeline.solve(
                job["upload"], out_dir,
                beam_time=job["beam_time"],
                concave=job["concave"],
                log=log,
            )
            job["result"] = result
            job["progress"] = 100
            job["status"] = "done"
            job["step"] = "完成"
            log_event(f"任务 {job['id']} 完成：{result.get('pieces')} 块，"
                      f"{result.get('steps', {}).get('total')}s，"
                      f"空洞 {result.get('metrics', {}).get('holes_pct')}% / "
                      f"重叠 {result.get('metrics', {}).get('overlap_pct')}%")
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["step"] = "失败"
            tb = traceback.format_exc()
            job["log"].append(tb[-2000:])
            # 错误也落盘：任务目录 logs/error.log，并记到服务器日志
            try:
                logs = Path(job["dir"]) / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                with open(logs / "error.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                             f"任务 {job['id']} 失败 =====\n{tb}\n")
            except Exception:
                pass
            log_event(f"任务 {job['id']} 失败：{exc}")
        finally:
            WORKER_LOCK.release()
            archive_job_log(job)
            try:
                if job.get("upload") and Path(job["upload"]).exists():
                    pass  # 保留 upload.jpg 供前端展示"原图/成品"对比
            except Exception:
                pass


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "JigsawWeb/1.0"

    def log_message(self, fmt, *args):        # 静音默认日志，只留自己打印的
        pass

    # ---------- 工具 ----------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _query(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    # ---------- GET ----------
    def do_GET(self):
        path = urlparse(self.path).path
        q = self._query()

        if path in ("/", "/index.html"):
            if not INDEX_FILE.exists():
                self._send(500, b"index.html missing", "text/plain; charset=utf-8")
                return
            self._send(200, INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/qr.png":
            try:
                import qrcode
                img = qrcode.make(f"http://{lan_ip()}:{self.server.server_port}/")
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                self._send(200, buf.getvalue(), "image/png")
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return

        if path == "/api/job":
            job_id = q.get("id", "")
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._json(404, {"error": "任务不存在或已过期"})
                return
            self._json(200, {
                "id": job["id"],
                "status": job["status"],
                "step": job["step"],
                "progress": job["progress"],
                "error": job["error"],
                "log": job["log"][-12:],
                "result": job["result"],
            })
            return

        if path == "/api/file":
            job_id = q.get("id", "")
            name = q.get("n", "")
            if name not in ALLOWED_FILES:
                self._json(403, {"error": "不允许访问该文件"})
                return
            base = Path(pipeline.job_root(), job_id)
            # .log 文件放在任务的 logs/ 子目录里
            target = (base / "logs" / name) if name.endswith(".log") else (base / name)
            if not target.exists():
                # 任务目录可能已被清理，退回去永久归档里找
                archived = log_dir() / f"job-{job_id}" / name
                if archived.exists():
                    target = archived
                else:
                    self._json(404, {"error": "文件不存在"})
                    return
            if name.endswith(".jpg"):
                ctype = "image/jpeg"
            elif name.endswith(".png"):
                ctype = "image/png"
            elif name.endswith(".log"):
                ctype = "text/plain; charset=utf-8"
            else:
                ctype = "application/json; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)
            return

        self._json(404, {"error": "not found"})

    # ---------- POST ----------
    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/solve":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"error": "没有收到图片数据"})
            return
        if length > MAX_UPLOAD:
            self._json(413, {"error": f"图片太大（上限 {MAX_UPLOAD // 1024 // 1024}MB）"})
            return

        data = self.rfile.read(length)
        if len(data) != length:
            self._json(400, {"error": "图片数据不完整"})
            return

        job = new_job()
        out_dir = Path(job["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        upload_path = out_dir / "upload.jpg"
        upload_path.write_bytes(data)

        # 校验能不能解码（顺便把手机 HEIC 之类的不支持格式挡在前面）
        import numpy as np
        import cv2
        probe = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if probe is None:
            self._json(400, {"error": "无法识别这张图片，请换成 JPG/PNG"})
            return

        job["upload"] = str(upload_path)
        job["beam_time"] = float(self.headers.get("X-Beam-Time") or 45)
        job["concave"] = (self.headers.get("X-Concave") or "1") != "0"
        JOB_QUEUE.put(job["id"])
        print(f"[任务 {job['id']}] 收到 {length // 1024}KB 图片，"
              f"beam={job['beam_time']}s concave={job['concave']}")
        log_event(f"任务 {job['id']} 收到 {length // 1024}KB 图片 "
                  f"beam={job['beam_time']}s concave={job['concave']}")
        self._json(200, {"id": job["id"]})


# --------------------------------------------------------------------------
# 启动
# --------------------------------------------------------------------------
def main() -> int:
    # 控制台强制 UTF-8，否则中文和二维码在 GBK 终端上会乱码 / 报错
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="拼图还原 · 局域网网页版")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    threading.Thread(target=worker, daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    ip = lan_ip()
    url = f"http://{ip}:{args.port}/"

    print("=" * 58)
    print("  拼图还原 · 局域网网页版 已启动")
    print("=" * 58)
    print(f"  本机访问 : http://127.0.0.1:{args.port}/")
    print(f"  手机访问 : {url}")
    print(f"  任务目录 : {pipeline.job_root()}")
    print(f"  日志目录 : {log_dir()}")
    print("-" * 58)
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print("  手机扫码打开上面的地址（同一 Wi-Fi 下）")
    except Exception:
        print("  （没装 qrcode，直接手输上面地址）")
    print("-" * 58)
    print("  Ctrl+C 停止")
    print("=" * 58, flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
