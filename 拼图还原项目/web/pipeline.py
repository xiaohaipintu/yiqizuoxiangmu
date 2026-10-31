"""拼图还原流水线封装 —— 把一张照片跑成"拼好的图"。

用法（命令行自测）：
    python web/pipeline.py --image input/pieces4.jpg --out C:/temp/job1

输出（--out 目录下）：
    upload.jpg          原始上传图（缩放后的副本）
    01_mask.png         背景抠除掩膜
    02_pieces.png       分割出的碎片
    41_union_rank.png   并集评估排名图（最左格为最优解）
    assembled.jpg       ★ 拼好的成品图（真实像素按位姿贴回）
    result.json         结果摘要（碎片数、空洞率、重叠率、各步耗时）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

WEB_DIR = Path(__file__).resolve().parent
PROJ_DIR = WEB_DIR.parent          # 拼图还原项目/
SRC_DIR = PROJ_DIR / "src"


def child_env() -> dict:
    """子进程强制 UTF-8：src/ 里的脚本会打印 px² 之类的字符，
    Windows 控制台默认 GBK 会直接 UnicodeEncodeError 崩掉。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def job_root() -> Path:
    """所有任务输出放在系统临时目录下：必须是纯 ASCII 路径，
    否则 Windows 版 cv2.imread 读不了中文路径。"""
    root = Path(tempfile.gettempdir()) / "jigsaw-jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------
# 基础 IO（统一走 imdecode/imencode，绕开 Windows 下 cv2 不支持中文路径的问题）
# --------------------------------------------------------------------------
def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不到图片: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray, quality: int = 92) -> None:
    suffix = Path(path).suffix.lower()
    params = []
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(suffix or ".png", img, params)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------
# 调用 src/ 下的脚本
# --------------------------------------------------------------------------
def _run(script: str, args: list, timeout: float, log, steps: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SRC_DIR / script)] + [str(a) for a in args]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJ_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=child_env(),
    )
    cost = time.time() - started
    if steps is not None:
        steps[script.replace(".py", "")] = round(cost, 2)
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-800:] + "\n" + (proc.stderr or "")[-800:]
        raise RuntimeError(f"{script} 失败 (rc={proc.returncode}):\n{tail.strip()}")
    if log:
        log(script, f"{cost:.1f}s")
    return proc


# --------------------------------------------------------------------------
# 成品图渲染：用真实像素 + 解出的刚体位姿把每块贴回原照片
# --------------------------------------------------------------------------
def render_assembled(
    image_path: str | Path,
    pieces_scaled_path: str | Path,
    best_union_path: str | Path,
    out_path: str | Path,
    margin: int = 28,
    max_side: int = 1400,
) -> bool:
    img = read_image(image_path)
    sol = json.loads(Path(best_union_path).read_text(encoding="utf-8"))
    placed = sol.get("placed") or []
    polys = sol.get("polys") or []
    if not placed or not polys:
        return False

    scaled = json.loads(Path(pieces_scaled_path).read_text(encoding="utf-8"))
    poly_by_id = {
        int(p["id"]): np.asarray(p["polygon"], dtype=np.float64)
        for p in scaled.get("pieces", [])
    }

    world = np.vstack([np.asarray(p, dtype=np.float64) for p in polys])
    mn = world.min(axis=0)
    mx = world.max(axis=0)
    width = int(math.ceil(mx[0] - mn[0])) + margin * 2
    height = int(math.ceil(mx[1] - mn[1])) + margin * 2
    width = max(width, 1)
    height = max(height, 1)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    off = np.array([mn[0] - margin, mn[1] - margin], dtype=np.float64)

    h_img, w_img = img.shape[:2]
    painted = 0
    for k, item in enumerate(placed):
        if k >= len(polys):
            break
        pid = int(item[0])
        poly_img = poly_by_id.get(pid)
        poly_world = np.asarray(polys[k], dtype=np.float64)
        if poly_img is None or len(poly_img) < 3 or len(poly_world) < 3:
            continue

        # 直接用「图片多边形 → 世界多边形」的顶点对拟合相似变换。
        # 不去猜 placed 里 rot/tx/ty 的符号约定，避免约定不一致时贴出一张白图。
        src = np.ascontiguousarray(poly_img, dtype=np.float32)
        dst = np.ascontiguousarray(poly_world, dtype=np.float32)
        mat, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if mat is None or not np.all(np.isfinite(mat)):
            continue

        # 拟合质量校验：顶点平均残差不能超过碎片尺度的 2%
        pred = cv2.transform(src.reshape(1, -1, 2), mat).reshape(-1, 2)
        resid = float(np.linalg.norm(pred - dst, axis=1).mean())
        scale = float(np.linalg.norm(np.ptp(poly_img, axis=0)))
        if scale > 0 and resid / scale > 0.02:
            continue

        # 世界坐标 -> 画布坐标：canvas = R·p_img + t - off
        mat_canvas = mat.copy()
        mat_canvas[:, 2] -= off

        # 只裁剪该碎片的源区域，避免对整张大图做 warp
        x0 = max(int(math.floor(poly_img[:, 0].min())) - 3, 0)
        y0 = max(int(math.floor(poly_img[:, 1].min())) - 3, 0)
        x1 = min(int(math.ceil(poly_img[:, 0].max())) + 3, w_img)
        y1 = min(int(math.ceil(poly_img[:, 1].max())) + 3, h_img)
        if x1 <= x0 or y1 <= y0:
            continue

        crop = img[y0:y1, x0:x1]
        # 裁剪坐标 = 图片坐标 - 裁剪原点，所以平移项要加上 R·(x0,y0)
        mat_crop = mat_canvas.copy()
        mat_crop[:, 2] += mat_crop[:, :2] @ np.array([x0, y0], dtype=np.float64)

        warped = cv2.warpAffine(
            crop, mat_crop, (width, height),
            flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255),
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        poly_canvas = poly_world - off
        cv2.fillPoly(mask, [np.round(poly_canvas).astype(np.int32)], 255)
        sel = mask > 0
        if not sel.any():
            continue
        canvas[sel] = warped[sel]
        painted += int(sel.sum())

    if painted == 0:
        return False

    if max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        canvas = cv2.resize(
            canvas,
            (max(int(width * scale), 1), max(int(height * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
    write_image(out_path, canvas, quality=93)
    return True


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def solve(
    image_path: str | Path,
    out_dir: str | Path,
    beam_time: float = 45.0,
    concave: bool = True,
    log=None,
) -> dict:
    """跑完整条流水线，返回结果摘要。out_dir 必须已存在且路径为 ASCII。"""
    out = Path(out_dir)
    if not str(out).isascii():
        raise RuntimeError(
            f"输出目录含非 ASCII 字符，cv2 在 Windows 下读不了：{out}\n"
            f"请换成纯英文路径，例如 {job_root()}"
        )
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    steps: dict[str, float] = {}

    def step(name: str, extra: str = ""):
        if log:
            log(name, extra)

    # 0) 归一化输入：转存一份到输出目录（cv2 在 Windows 下读不了中文路径），并限制尺寸
    src_img = read_image(image_path)
    h, w = src_img.shape[:2]
    max_side = 2400
    if max(h, w) > max_side:
        ratio = max_side / float(max(h, w))
        src_img = cv2.resize(src_img, (int(w * ratio), int(h * ratio)),
                             interpolation=cv2.INTER_AREA)
    upload = out / "upload.jpg"
    write_image(upload, src_img, quality=92)
    image_path = upload

    # 1) 分割
    step("segment", "抠出碎片…")
    seg_args = ["--image", image_path, "--out", out]
    if concave:
        seg_args.append("--concave")
    _run("segment.py", seg_args, timeout=180, log=step, steps=steps)
    pieces_json = out / "pieces.json"
    if not pieces_json.exists():
        raise RuntimeError("分割失败：没有生成 pieces.json")
    n_pieces = len(json.loads(pieces_json.read_text(encoding="utf-8")).get("pieces", []))
    if n_pieces < 2:
        raise RuntimeError(f"只找到 {n_pieces} 块碎片，照片里可能没有可区分的碎片")

    # 2) 尺度补偿
    step("rescale", f"{n_pieces} 块，尺度补偿…")
    scaled_json = out / "pieces_scaled.json"
    _run("rescale_pieces.py",
         ["--pieces", pieces_json, "--out", scaled_json], timeout=120, log=step, steps=steps)

    # 3) 束搜索拼合
    step("beam", f"搜索拼法（最多 {int(beam_time)} 秒）…")
    _run("beam_search.py",
         ["--pieces", scaled_json, "--out", out,
          "--beam", "300", "--time", str(float(beam_time))],
         timeout=beam_time + 240, log=step, steps=steps)
    sol_json = out / "beam_solutions.json"
    if not sol_json.exists():
        raise RuntimeError("搜索失败：没有生成 beam_solutions.json")
    n_sol = json.loads(sol_json.read_text(encoding="utf-8")).get("n", 0)
    if not n_sol:
        raise RuntimeError("没有找到任何拼法，换个背景更干净的照片试试")

    # 4) 并集评估
    step("eval", f"{n_sol} 个候选，评估空洞/重叠…")
    _run("eval_union.py",
         ["--solutions", sol_json, "--out", out, "--top", "4"], timeout=180, log=step, steps=steps)

    # 5) 渲染成品图
    step("render", "按位姿贴回真实像素…")
    assembled = out / "assembled.jpg"
    best_union = out / "best_union.json"
    metrics: dict = {}
    if best_union.exists():
        try:
            best = json.loads(best_union.read_text(encoding="utf-8"))
            metrics = {
                "holes_pct": round(float(best.get("holes_pct", 0.0)), 3),
                "overlap_pct": round(float(best.get("overlap_pct", 0.0)), 3),
            }
        except Exception:
            metrics = {}
    rendered = False
    if best_union.exists():
        try:
            rendered = render_assembled(image_path, scaled_json, best_union, assembled)
        except Exception as exc:                      # 渲染失败退回排名图
            metrics["render_error"] = str(exc)
    if not rendered:
        rank = out / "41_union_rank.png"
        if rank.exists():
            write_image(assembled, read_image(rank), quality=92)
        else:
            raise RuntimeError("渲染成品图失败")

    steps["total"] = round(time.time() - t_start, 2)
    result = {
        "pieces": n_pieces,
        "solutions": n_sol,
        "metrics": metrics,
        "steps": steps,
        "files": [p.name for p in sorted(out.iterdir()) if p.is_file()],
    }
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    step("done", "完成")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="拼图还原流水线")
    ap.add_argument("--image", required=True, help="输入照片")
    ap.add_argument("--out", required=True, help="输出目录（ASCII 路径）")
    ap.add_argument("--time", type=float, default=45.0, help="束搜索时间预算（秒）")
    ap.add_argument("--no-concave", action="store_true", help="关闭凹形拟合")
    args = ap.parse_args()

    def log(name, extra=""):
        print(f"[{name}] {extra}")

    out_dir = args.out
    if not str(out_dir).isascii():
        out_dir = str(job_root() / Path(out_dir).name)
        print(f"[提示] 输出目录非英文路径，改用 {out_dir}")

    result = solve(
        args.image, out_dir,
        beam_time=args.time,
        concave=not args.no_concave,
        log=log,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
