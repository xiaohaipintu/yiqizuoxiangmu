"""任务状态落盘：让任务在「关掉网页 / 服务重启」之后还能查到。

原来任务只存在 `server.py` 的内存字典里，页面一刷新、服务一重启就查不到了。
这里把每个任务的关键状态写到 `<job_dir>/job.json`：

    {"id","status","step","progress","error","created","updated","result"}

落盘时机：建任务、每推进一步、完成、失败。
读取方式：`list_jobs()` 扫任务目录，`load()` 按 id 读单个。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

KEEP = ("id", "status", "step", "progress", "error", "created", "result")


def _dir(root, job_id: str) -> Path:
    return Path(root) / job_id


def save(root, job: dict) -> None:
    """把任务状态写进 job.json（字段少而稳，失败也不影响主流程）。"""
    try:
        d = _dir(root, job["id"])
        d.mkdir(parents=True, exist_ok=True)
        payload = {k: job.get(k) for k in KEEP}
        payload["updated"] = time.time()
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "job.json")            # 原子替换，避免读到半个文件
    except Exception:
        pass


def load(root, job_id: str) -> dict | None:
    p = _dir(root, job_id) / "job.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_jobs(root, limit: int = 60) -> list[dict]:
    """按创建时间倒序列出任务（只读 job.json，不碰内存）。"""
    base = Path(root)
    if not base.exists():
        return []
    out = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        j = load(root, d.name)
        if not j:
            continue
        j["thumbnail"] = _thumbnail(d)
        out.append(j)
    out.sort(key=lambda j: (j.get("created") or 0), reverse=True)
    return out[:limit]


def _thumbnail(d: Path) -> str | None:
    """挑一张能代表当前进度的图（前端直接拿去显示）。"""
    for name in ("assembled_labels.jpg", "assembled.jpg",
                 "41_union_rank.png", "beam_progress.png",
                 "keyed.jpg", "upload.jpg"):
        if (d / name).exists():
            return name
    return None


def summarize(job: dict) -> dict:
    """给列表用的精简摘要。"""
    r = job.get("result") or {}
    m = r.get("metrics") or {}
    st = r.get("steps") or {}
    return {
        "pieces": r.get("pieces"),
        "solutions": r.get("solutions"),
        "holes_pct": m.get("holes_pct"),
        "overlap_pct": m.get("overlap_pct"),
        "total_s": st.get("total"),
        "candidates": len(r.get("candidates") or []),
        "texture_mean_dE": (r.get("texture") or {}).get("mean_dE"),
    }
