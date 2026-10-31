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


def write_step_log(logs_dir: Path, script: str, cmd: list, proc, cost: float) -> None:
    """把每一步的完整命令、stdout、stderr 落盘。

    - `logs/<脚本名>.log`：这一次的完整内容
    - `logs/run.log`：整个任务的追加流水（含时间戳）
    """
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        body = (
            f"\n===== {stamp}  {script}  (rc={proc.returncode}, {cost:.2f}s) =====\n"
            f"$ {' '.join(str(c) for c in cmd)}\n"
            f"--- stdout ---\n{proc.stdout or '(空)'}\n"
            f"--- stderr ---\n{proc.stderr or '(空)'}\n"
        )
        (logs_dir / f"{script.replace('.py', '')}.log").write_text(body, encoding="utf-8")
        with open(logs_dir / "run.log", "a", encoding="utf-8") as fh:
            fh.write(body)
    except Exception:
        pass          # 日志写不进去不能影响主流程


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
# 命名与标注：碎片叫 A / B / C…，每块碎片的边叫 A1 / A2 / …
# --------------------------------------------------------------------------
_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def piece_name(index: int) -> str:
    """0 → A, 25 → Z, 26 → AA …（按 pieces.json 里的顺序）"""
    out = ""
    i = index + 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        out = _ALPHA[r] + out
    return out


def edge_start_index(poly) -> int:
    """边的编号起点：最上方的顶点（同高取最左）。"""
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return 0
    return min(range(n), key=lambda k: (round(float(pts[k][1]), 2),
                                        round(float(pts[k][0]), 2)))


def edges_of(poly, start: int | None = None) -> list:
    """返回 [(边序号, 起点, 终点)]。

    编号规则（固定、可复现）：从 start 顶点开始（默认取最上方顶点，同高取最左），
    按多边形顶点顺序环状编号，第 k 条边 = 第 k 个顶点 → 第 k+1 个顶点。

    ⚠️ 拼好的图上必须传入**原图**算出的 start，否则碎片被旋转后
    「最上方顶点」会变，同一个名字就指到别的边上了。
    """
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return []
    if start is None:
        start = edge_start_index(pts)
    out = []
    for k in range(n):
        i = (start + k) % n
        j = (i + 1) % n
        out.append((k + 1, pts[i], pts[j]))
    return out


def _text(img, s: str, org, size: float, color, thick: int) -> None:
    """以 org 为中心写一行字（先描黑边，保证在任何底色上都看得清）。"""
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    x = int(round(org[0] - w / 2))
    y = int(round(org[1] + h / 2))
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                color, thick, cv2.LINE_AA)


def draw_labels(img: np.ndarray, pieces: list) -> np.ndarray:
    """在图上标出碎片名（质心）与边名（边中点，略往内部偏）。

    pieces: [(碎片名, 多边形坐标数组[, 边编号起点顶点索引]), …]
    第三个元素必须传「原图」上算出的起点，这样成品图上的边名和原图一一对应。
    """
    if not pieces:
        return img
    scale = max(0.45, img.shape[0] / 950.0)      # 字号随图片高度缩放
    for item in pieces:
        name, poly = item[0], item[1]
        start = item[2] if len(item) > 2 else None
        pts = np.asarray(poly, dtype=np.float64)
        if len(pts) < 3:
            continue
        cv2.polylines(
            img,
            [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
            True, (248, 189, 56), max(1, int(round(2 * scale))), cv2.LINE_AA,
        )
        centroid = pts.mean(axis=0)
        _text(img, name, centroid, 1.15 * scale, (255, 255, 255),
              max(2, int(round(2 * scale))))
        for no, a, b in edges_of(pts, start):
            mid = (a + b) / 2.0
            d = mid - centroid
            length = float(np.linalg.norm(d))
            if length > 1e-6:                     # 往碎片内部挪一点，别压在轮廓上
                mid = mid - d / length * (16 * scale)
            _text(img, f"{name}{no}", mid, 0.62 * scale, (138, 240, 254),
                  max(1, int(round(1.6 * scale))))
    return img


# --------------------------------------------------------------------------
# 拼合关系：算出来哪条边和哪条边真的贴在一起（对照表用）
# --------------------------------------------------------------------------
def _share_on_segment(a1, b1, a2, b2, tol: float) -> float:
    """a1-b1 上有多少比例的采样点落在 a2-b2 上（tol 像素内）。"""
    seg = b1 - a1
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        return 0.0
    other = b2 - a2
    other_len2 = float(other @ other)
    if other_len2 < 1e-9:
        return 0.0
    ts = np.linspace(0.0, 1.0, max(int(length / 2.0), 8))
    pts = a1[None, :] + seg[None, :] * ts[:, None]
    t = np.clip(((pts - a2[None, :]) @ other) / other_len2, 0.0, 1.0)
    proj = a2[None, :] + t[:, None] * other[None, :]
    dist = np.linalg.norm(pts - proj, axis=1)
    return float((dist <= tol).mean())


def find_joints(pieces: list, tol: float = 3.0, share: float = 0.3) -> list:
    """找出真正贴合的边对。

    pieces: [(碎片名, 多边形, 边编号起点顶点索引), …]
    返回 [{"a": "A1", "b": "C4", "share": 100.0}]，按贴合比例降序。
    """
    edges = []          # (边名, 起点, 终点, 碎片名)
    for item in pieces:
        name, poly = item[0], item[1]
        start = item[2] if len(item) > 2 else None
        for no, a, b in edges_of(poly, start):
            edges.append((f"{name}{no}", np.asarray(a, float), np.asarray(b, float), name))

    out = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            n1, a1, b1, p1 = edges[i]
            n2, a2, b2, p2 = edges[j]
            if p1 == p2:
                continue
            best = max(
                _share_on_segment(a1, b1, a2, b2, tol),
                _share_on_segment(a2, b2, a1, b1, tol),
            )
            if best >= share:
                out.append({"a": n1, "b": n2, "share": round(best * 100, 1)})
    return sorted(out, key=lambda x: -x["share"])


# --------------------------------------------------------------------------
# 候选解重排：eval_union 只看空洞/重叠，分不出「接错了边」
# --------------------------------------------------------------------------
def eval_metrics(polys: list) -> dict:
    """用 src/eval_union.py 的 evaluate() 算空洞/重叠/凸包空隙。"""
    try:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        import eval_union  # noqa: E402
        ps = [np.asarray(p, dtype=float) for p in polys]
        m = eval_union.evaluate(ps)
        out = {
            "holes_pct": round(float(m["holes_pct"]), 3),
            "overlap_pct": round(float(m["overlap_pct"]), 3),
        }
        try:
            out["hull_gap_pct"] = round(float(eval_union.hull_gap(ps)), 3)
        except Exception:
            pass
        return out
    except Exception:
        return {}


def _joint_length_diff(joints: list, lengths: dict) -> float:
    if not joints:
        return 999.0
    diffs = []
    for j in joints:
        la = lengths.get(j["a"])
        lb = lengths.get(j["b"])
        if not la or not lb:
            continue
        diffs.append(abs(la - lb) / max(la, lb) * 100.0)
    return round(float(np.mean(diffs)), 2) if diffs else 999.0


def rerank_solutions(sol_json: Path, names: dict, starts: dict,
                     share: float = 0.9) -> dict | None:
    """在 beam 给出的候选里挑「边接得最吻合」的那个。

    eval_union.py 只按空洞/重叠排序，而错误的拼法同样是 0 空洞 0 重叠
    （碎片错边相接一样能拼满），所以必须再按**对接边长度一致度**重排。

    返回 {"entry": 候选解, "joints": [...], "diff": 平均长度差%, "index": i}
    """
    try:
        doc = json.loads(Path(sol_json).read_text(encoding="utf-8"))
        sols = doc.get("solutions") or []
    except Exception:
        return None
    best = None
    for idx, entry in enumerate(sols):
        placed = entry.get("placed") or []
        polys = entry.get("polys") or []
        if not placed or len(placed) != len(polys):
            continue
        labeled = [(names.get(int(item[0]), str(item[0])),
                    np.asarray(polys[k], dtype=np.float64),
                    starts.get(int(item[0]))) for k, item in enumerate(placed)]
        joints = find_joints(labeled, share=share)
        pieces = set()
        for j in joints:
            pieces.add(j["a"].rstrip("0123456789"))
            pieces.add(j["b"].rstrip("0123456789"))
        if len(pieces) < len(names):        # 有碎片没接上，直接淘汰
            continue
        lengths = {}
        for k, item in enumerate(placed):
            pid = int(item[0])
            poly = poly_by_id_edges = np.asarray(polys[k], dtype=np.float64)
            for no, a, b in edges_of(poly, starts.get(pid)):
                lengths[f"{names.get(pid, str(pid))}{no}"] = float(
                    np.linalg.norm(np.asarray(b) - np.asarray(a)))
        diff = _joint_length_diff(joints, lengths)
        cand = {"entry": entry, "joints": joints, "diff": diff,
                "index": idx, "pairs": len(joints)}
        # 关键：长度一致度优先，其次贴合对数多的
        key = (diff, -len(joints))
        if best is None or key < best["_key"]:
            cand["_key"] = key
            best = cand
    return best


# --------------------------------------------------------------------------
# 调用 src/ 下的脚本
# --------------------------------------------------------------------------
def _run(script: str, args: list, timeout: float, log, steps: dict | None = None,
         logs_dir: Path | None = None) -> subprocess.CompletedProcess:
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
    if logs_dir is not None:
        write_step_log(logs_dir, script, cmd, proc, cost)
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
    names: dict | None = None,
    labels_out: str | Path | None = None,
    edge_starts: dict | None = None,
    info: dict | None = None,
    solution: dict | None = None,
    fg_mask_path: str | Path | None = None,
) -> bool:
    img = read_image(image_path)
    if solution is not None:
        sol = solution
    else:
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
    canvas_polys = []

    # 前景掩膜：只贴真实碎片像素，避免把碎片边缘那一圈背景色也贴上去
    # （否则每块都自带一条绿色包边，拼缝看起来就像没拼上）
    fg = None
    if fg_mask_path:
        try:
            fg = read_image(fg_mask_path)
            if fg.ndim == 3:
                fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
            if fg.shape[:2] != (h_img, w_img):
                fg = None
        except Exception:
            fg = None
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
        if fg is not None:
            fg_warped = cv2.warpAffine(
                fg[y0:y1, x0:x1], mat_crop, (width, height),
                flags=cv2.INTER_NEAREST, borderValue=0,
            )
            # 外扩 1px：宁可轻微重叠，也不要露出白缝
            fg_warped = cv2.dilate(fg_warped, np.ones((3, 3), np.uint8))
            sel = sel & (fg_warped > 127)
        if not sel.any():
            continue
        canvas[sel] = warped[sel]
        painted += int(sel.sum())
        canvas_polys.append((pid, poly_canvas))

    if painted == 0:
        return False

    if max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        canvas = cv2.resize(
            canvas,
            (max(int(width * scale), 1), max(int(height * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
        canvas_polys = [(pid, poly * scale) for pid, poly in canvas_polys]
    write_image(out_path, canvas, quality=93)

    # 带名称的版本：碎片名（A/B/C）+ 边名（A1/A2…）
    # ⚠️ 边编号起点必须沿用**原图**算出的顶点索引，否则碎片在成品图上被旋转后
    #    「最上方顶点」变了，同一个名字会指到别的边上去（曾导致 A1 被误读成 B6）
    labeled = [
        (names.get(pid, str(pid)) if names else str(pid),
         poly,
         (edge_starts or {}).get(pid))
        for pid, poly in canvas_polys
    ]
    if labels_out:
        write_image(labels_out, draw_labels(canvas.copy(), labeled), quality=93)
    if info is not None:
        info["joints"] = find_joints(labeled)
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

    # 日志目录：每步的命令行 / stdout / stderr 全落盘，任务结束也不会丢
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "run.log", "a", encoding="utf-8") as fh:
        fh.write(
            "\n" + "#" * 70 + "\n"
            f"# 任务开始 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# 输入 {image_path}\n"
            f"# 参数 beam_time={beam_time}s concave={concave}\n"
            f"# Python {sys.version.split()[0]}  cwd={PROJ_DIR}\n"
            + "#" * 70 + "\n"
        )

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
    _run("segment.py", seg_args, timeout=180, log=step, steps=steps,
         logs_dir=logs)
    pieces_json = out / "pieces.json"
    if not pieces_json.exists():
        raise RuntimeError("分割失败：没有生成 pieces.json")
    n_pieces = len(json.loads(pieces_json.read_text(encoding="utf-8")).get("pieces", []))
    if n_pieces < 2:
        raise RuntimeError(f"只找到 {n_pieces} 块碎片，照片里可能没有可区分的碎片")

    # 1.5) 给碎片和边命名：A/B/C… 与 A1/A2/…（原图与成品图用同一套编号）
    piece_list = json.loads(pieces_json.read_text(encoding="utf-8")).get("pieces", [])
    names: dict[int, str] = {}
    starts: dict[int, int] = {}          # 每块碎片的边编号起点顶点（必须原图算一次、全程复用）
    labels: list[dict] = []
    for i, piece in enumerate(piece_list):
        pid = int(piece["id"])
        name = piece_name(i)
        names[pid] = name
        poly = np.asarray(piece["polygon"], dtype=np.float64)
        starts[pid] = edge_start_index(poly)
        labels.append({
            "name": name,
            "id": pid,
            "vertices": int(len(poly)),
            "area": round(float(piece.get("area") or 0.0), 1),
            "edges": [
                {
                    "name": f"{name}{no}",
                    "length": round(float(np.linalg.norm(b - a)), 1),
                }
                for no, a, b in edges_of(poly, starts[pid])
            ],
        })
    # 原图上的标注版
    upload_labels = out / "upload_labels.jpg"
    try:
        marked = src_img.copy()
        draw_labels(marked, [(names[int(p["id"])],
                              np.asarray(p["polygon"], float),
                              starts[int(p["id"])])
                             for p in piece_list])
        write_image(upload_labels, marked, quality=92)
    except Exception:
        upload_labels = None

    # 2) 尺度补偿
    step("rescale", f"{n_pieces} 块，尺度补偿…")
    scaled_json = out / "pieces_scaled.json"
    _run("rescale_pieces.py",
         ["--pieces", pieces_json, "--out", scaled_json], timeout=120, log=step, steps=steps,
         logs_dir=logs)

    # 3) 束搜索拼合
    step("beam", f"搜索拼法（最多 {int(beam_time)} 秒）…")
    _run("beam_search.py",
         ["--pieces", scaled_json, "--out", out,
          "--beam", "300", "--time", str(float(beam_time))],
         timeout=beam_time + 240, log=step, steps=steps,
         logs_dir=logs)
    sol_json = out / "beam_solutions.json"
    if not sol_json.exists():
        raise RuntimeError("搜索失败：没有生成 beam_solutions.json")
    n_sol = json.loads(sol_json.read_text(encoding="utf-8")).get("n", 0)
    if not n_sol:
        raise RuntimeError("没有找到任何拼法，换个背景更干净的照片试试")

    # 4) 并集评估
    step("eval", f"{n_sol} 个候选，评估空洞/重叠…")
    _run("eval_union.py",
         ["--solutions", sol_json, "--out", out, "--top", "4"], timeout=180, log=step, steps=steps,
         logs_dir=logs)

    # 5) 渲染成品图
    step("render", "按位姿贴回真实像素…")
    assembled = out / "assembled.jpg"
    assembled_labels = out / "assembled_labels.jpg"
    best_union = out / "best_union.json"

    # 4.5) 候选重排：eval_union 只按空洞/重叠排序，分不出「接错了边」
    #      （错边相接同样能拼满、同样是 0 空洞 0 重叠），所以再按对接边长度一致度挑一遍
    step("rerank", "按对接边长度一致度重排候选…")
    solution = None
    ranking: dict = {}
    pick = None
    try:
        pick = rerank_solutions(sol_json, names, starts)
    except Exception as exc:
        ranking["error"] = str(exc)
    if pick:
        entry = pick["entry"]
        solution = {"placed": entry.get("placed"), "polys": entry.get("polys")}
        ranking = {
            "chosen": f"beam#{pick['index']}",
            "mean_len_diff_pct": pick["diff"],
            "joint_pairs": pick["pairs"],
        }
        (out / "chosen_solution.json").write_text(
            json.dumps({"ranking": ranking, **solution}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif best_union.exists():
        try:
            solution = json.loads(best_union.read_text(encoding="utf-8"))
            ranking = {"chosen": "best_union"}
        except Exception:
            solution = None

    metrics: dict = {}
    if solution and solution.get("polys"):
        metrics = eval_metrics(solution["polys"])
    if not metrics and best_union.exists():
        try:
            best = json.loads(best_union.read_text(encoding="utf-8"))
            metrics = {
                "holes_pct": round(float(best.get("holes_pct", 0.0)), 3),
                "overlap_pct": round(float(best.get("overlap_pct", 0.0)), 3),
            }
        except Exception:
            metrics = {}

    rendered = False
    info: dict = {}
    if solution:
        try:
            rendered = render_assembled(image_path, scaled_json, best_union, assembled,
                                        names=names, labels_out=assembled_labels,
                                        edge_starts=starts, info=info,
                                        solution=solution,
                                        fg_mask_path=out / "01_mask.png")
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
        "labels": labels,
        "joints": info.get("joints", []),
        "ranking": ranking,
        "has_labels": bool(upload_labels),
        "files": [p.name for p in sorted(out.iterdir()) if p.is_file()],
    }
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(logs / "run.log", "a", encoding="utf-8") as fh:
        fh.write(
            "\n" + "=" * 70 + "\n"
            f"# 任务结束 {time.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"总耗时 {steps.get('total', 0)}s\n"
            + "=" * 70 + "\n"
            + json.dumps(result, ensure_ascii=False, indent=2) + "\n"
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
