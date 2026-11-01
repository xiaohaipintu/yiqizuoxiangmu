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


# --------------------------------------------------------------------------
# 清理多边形：影子 / 阴影会把一条直边切成两段，合并近共线的假顶点
# --------------------------------------------------------------------------
CLEAN_ANGLE_TOL = 12.0       # 内角与 180° 差 <12° 就当成同一条直线（实测假顶点 170°）


def _interior_angle(a, b, c) -> float:
    """顶点 b 处的内角（度）。直边=180°，直角=90°。"""
    v1, v2 = np.asarray(a, float) - np.asarray(b, float), np.asarray(c, float) - np.asarray(b, float)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0))))


def simplify_polygon(poly, tol: float = CLEAN_ANGLE_TOL,
                     min_edge_frac: float = 0.02) -> np.ndarray:
    """去掉「几乎是直线」的顶点（影子、毛刺、抗锯齿造成的假角），再剔除碎边。

    两件事分开做：
    1. **共线合并**：内角与 180° 差 < tol 的顶点删掉（影子把直边切两段时会这样）；
    2. **碎边剔除**：比周长 min_edge_frac 还短的边（亚像素拟合留下的毛刺尖刺）合并掉。
    """
    pts = [np.asarray(p, dtype=np.float64) for p in poly]
    # 1) 共线合并
    for _ in range(8):
        n = len(pts)
        if n <= 3:
            break
        keep = [abs(180.0 - _interior_angle(pts[k - 1], pts[k], pts[(k + 1) % n])) >= tol
                for k in range(n)]
        if all(keep):
            break
        pts = [p for p, kp in zip(pts, keep) if kp]
        if not pts:
            break
    # 2) 碎边剔除（至少保留 3 个顶点）
    for _ in range(8):
        n = len(pts)
        if n <= 3:
            break
        per = float(sum(np.linalg.norm(pts[(k + 1) % n] - pts[k]) for k in range(n)))
        cut = per * min_edge_frac
        drop = set()
        for k in range(n):
            if k in drop:
                continue
            nxt = (k + 1) % n
            if np.linalg.norm(pts[nxt] - pts[k]) < cut:
                drop.add(nxt if np.linalg.norm(pts[nxt] - pts[k]) < cut else k)
        if not drop:
            break
        keep_idx = [k for k in range(n) if k not in drop]
        if len(keep_idx) < 3:
            break
        pts = [pts[k] for k in keep_idx]
    return np.asarray(pts, dtype=np.float64)


def _rebuild_piece(piece: dict, poly: np.ndarray) -> dict:
    """按简化后的多边形重算面积/质心/边表（字段名对齐 segment.py 的输出）。"""
    n = len(poly)
    edges = []
    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        d = b - a
        edges.append({
            "index": k,
            "a": [float(a[0]), float(a[1])],
            "b": [float(b[0]), float(b[1])],
            "length": float(np.linalg.norm(d)),
            "angle": float(math.degrees(math.atan2(float(d[1]), float(d[0])))),
            "mid": [float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)],
        })
    angles = [
        _interior_angle(poly[k - 1], poly[k], poly[(k + 1) % n]) for k in range(n)
    ]
    perimeter = float(sum(e["length"] for e in edges))
    piece = dict(piece)
    piece["polygon"] = [[float(x), float(y)] for x, y in poly]
    piece["edges"] = edges
    piece["interior_angles"] = angles
    piece["area"] = abs(float(cv2.contourArea(poly.astype(np.float32))))
    piece["centroid"] = [float(poly[:, 0].mean()), float(poly[:, 1].mean())]
    piece["perimeter"] = perimeter
    piece["bbox"] = [float(poly[:, 0].min()), float(poly[:, 1].min()),
                     float(poly[:, 0].max()), float(poly[:, 1].max())]
    piece.pop("edge_fits", None)      # 边表变了，旧的拟合信息作废
    return piece


def clean_pieces(pieces_json: Path, tol: float = CLEAN_ANGLE_TOL) -> dict:
    """把 pieces.json 里的多边形做共线合并，原地写回。返回统计。"""
    try:
        doc = json.loads(Path(pieces_json).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}
    before = after = 0
    changed = []
    for i, piece in enumerate(doc.get("pieces", [])):
        poly = np.asarray(piece.get("polygon") or [], dtype=np.float64)
        if len(poly) < 4:
            continue
        before += len(poly)
        simple = simplify_polygon(poly, tol)
        after += len(simple)
        if len(simple) < len(poly):
            changed.append({"piece": piece_name(i), "from": len(poly), "to": len(simple)})
            doc["pieces"][i] = _rebuild_piece(piece, simple)
    if changed:
        Path(pieces_json).write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"vertices_before": before, "vertices_after": after, "merged": changed}


GOOD_LEN_DIFF = 4.0          # 对接边长差 ≤4% 认为拼对了GOOD_PAIRS_RATIO = 1.0       # 贴合对数至少要 n-1（串成一串）


def beam_plan(n_pieces: int, total_edges: int) -> list:
    """按碎片数量和边的总数，自动排一个「先小预算试，不行再加时」的计划。

    用户不需要选搜索时长：先给一个够用的预算，拼得不够吻合再翻倍重搜，
    最多三次（总时长在 solve() 里还有 150 秒硬上限）。
    """
    base = {2: 5, 3: 8, 4: 18, 5: 35, 6: 50}.get(n_pieces, 70)
    avg_edges = total_edges / max(n_pieces, 1)
    if avg_edges > 8:            # 凹形/多顶点碎片，搜索空间更大
        base = int(base * 1.4)
    base = max(5, min(base, 70))
    return [base, min(base * 2, 110), min(int(base * 3.5), 150)]


def beam_ok(pick: dict | None, n_pieces: int) -> bool:
    """这次搜索的结果够不够好（够好就不再加时）。"""
    if not pick:
        return False
    if pick["pairs"] < max(1, n_pieces - 1):
        return False
    return pick["diff"] <= GOOD_LEN_DIFF


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


def solution_metrics(entry: dict, names: dict, starts: dict, share: float = 0.9) -> dict | None:
    """给一个候选解打分：对接边长差 / 凸包空隙 / 空洞 / 重叠 / 矩形贴合度 / 轮廓复杂度。"""
    placed = entry.get("placed") or []
    polys = entry.get("polys") or []
    if not placed or len(placed) != len(polys):
        return None
    labeled = [(names.get(int(item[0]), str(item[0])),
                np.asarray(polys[k], dtype=np.float64),
                starts.get(int(item[0]))) for k, item in enumerate(placed)]

    joints = find_joints(labeled, share=share)
    joined = set()
    for j in joints:
        joined.add(j["a"].rstrip("0123456789"))
        joined.add(j["b"].rstrip("0123456789"))
    if len(joined) < len(names):          # 有碎片没接上，不算完整拼法
        return None

    lengths = {}
    for name, poly, st in labeled:
        for no, a, b in edges_of(poly, st):
            lengths[f"{name}{no}"] = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    diff = _joint_length_diff(joints, lengths)

    arrs = [np.asarray(p, dtype=float) for p in polys]
    info = {"len_diff_pct": diff, "joint_pairs": len(joints)}
    try:
        from shapely.geometry import Polygon as _SPoly
        from shapely.ops import unary_union as _union
        import cv2 as _cv2
        ps = [_SPoly(p).buffer(0) for p in arrs]
        u = _union(ps)
        total = sum(p.area for p in ps)
        holes = sum(abs(_SPoly(r).area) for r in getattr(u, "interiors", [])) / total * 100
        overlap = (total - u.area) / total * 100
        hull = abs(float(_cv2.contourArea(_cv2.convexHull(
            np.vstack(arrs).astype(np.float32)))))
        hull_gap = max(0.0, hull - total) / total * 100
        rect = u.minimum_rotated_rectangle
        rect_fill = u.area / rect.area * 100 if rect.area > 0 else 0.0
        outline_v = len(getattr(u, "exterior", None).coords) - 1 if hasattr(u, "exterior") else 0
        info.update({
            "hull_gap_pct": round(hull_gap, 2),
            "holes_pct": round(holes, 3),
            "overlap_pct": round(overlap, 3),
            "rect_fill_pct": round(rect_fill, 1),
            "outline_vertices": outline_v,
        })
    except Exception:
        info.setdefault("hull_gap_pct", 0.0)

    # 综合分：越小越"合理"。矩形先验是软约束 —— 只有几乎贴满矩形(≥97%)才大幅加分，
    # 这样非矩形原图不会被硬拽成矩形，但真有矩形解时它会脱颖而出。
    rect_fill = info.get("rect_fill_pct")
    rect_pen = 0.0 if rect_fill is None or rect_fill >= 97 else 6.0 + 0.08 * (97 - rect_fill)
    score = (info["len_diff_pct"]
             + 2.0 * info.get("hull_gap_pct", 0.0)
             + 3.0 * max(0.0, info.get("holes_pct", 0.0))
             + 3.0 * max(0.0, info.get("overlap_pct", 0.0))
             + rect_pen
             + 0.2 * max(0, info.get("outline_vertices", 4) - 4))
    info["rect_penalty"] = round(rect_pen, 2)
    info["score"] = round(float(score), 2)
    return info


def rerank_solutions(sol_json: Path, names: dict, starts: dict,
                     share: float = 0.9, top_k: int = 5,
                     extra_sols: list | None = None) -> dict | None:
    """在 beam 给出的候选里，按「拼法的合理性」排序，返回最优 + 前 top_k 名。

    只看空洞/重叠不够 —— 错边相接同样是 0 空洞 0 重叠。这里综合：
    对接边长度差、凸包空隙、空洞/重叠、矩形贴合度、轮廓顶点数。
    extra_sols 用来并入 T 型接缝搜索（split_search）的候选。
    """
    try:
        doc = json.loads(Path(sol_json).read_text(encoding="utf-8"))
        sols = list(doc.get("solutions") or [])
    except Exception:
        sols = []
    if extra_sols:
        sols.extend(extra_sols)
    if not sols:
        return None

    ranked = []
    for idx, entry in enumerate(sols):
        info = solution_metrics(entry, names, starts, share=share)
        if info is None:
            continue
        ranked.append({"index": idx, "entry": entry, "metrics": info,
                       "score": info["score"]})
    if not ranked:
        return None
    ranked.sort(key=lambda r: (r["score"], r["index"]))

    # 去重：不同的"放置顺序"可能得到完全一样的拼法（顶点对称），合并掉
    uniq, seen = [], set()
    for r in ranked:
        m = r["metrics"]
        sig = (round(m.get("len_diff_pct", 0), 1),
               round(m.get("hull_gap_pct", 0), 1),
               round(m.get("rect_fill_pct", 0), 0),
               m.get("outline_vertices"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(r)
    ranked = uniq

    best = ranked[0]
    return {
        "entry": best["entry"],
        "index": best["index"],
        "diff": best["metrics"]["len_diff_pct"],
        "pairs": best["metrics"]["joint_pairs"],
        "score": best["score"],
        "metrics": best["metrics"],
        "top": ranked[:top_k],
    }


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


def _run_soft(script: str, args: list, timeout: float,
              logs_dir: Path | None = None) -> subprocess.CompletedProcess | None:
    """跑一个「可选」步骤：失败只记日志，不中断整条流水线。"""
    cmd = [sys.executable, str(SRC_DIR / script)] + [str(a) for a in args]
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJ_DIR), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=child_env())
    except Exception:
        return None
    try:
        if logs_dir is not None:
            write_step_log(logs_dir, script, cmd, proc, time.time() - started)
    except Exception:
        pass
    return proc


# --------------------------------------------------------------------------
# 先验约束：1 个凹形 + 3 个三角形，每个三角形只能对应一个缺口（双射）
# --------------------------------------------------------------------------
def _reflex_count(poly) -> int:
    n = len(poly)
    sa = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        sa += x1 * y2 - x2 * y1
    sign = 1.0 if sa > 0 else -1.0
    c = 0
    for i in range(n):
        o, a, b = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        if cross * sign < 0:
            c += 1
    return c


def notch_applicable(pieces_json: Path) -> tuple[bool, dict]:
    """碎片集合是否满足「1 个凹形(3 凹顶点) + 恰好 3 个三角形」。"""
    try:
        doc = json.loads(Path(pieces_json).read_text(encoding="utf-8"))
    except Exception:
        return False, {}
    pieces = doc.get("pieces") or []
    concave = [(p, _reflex_count(p["polygon"])) for p in pieces]
    concave = [(p, k) for p, k in concave if k > 0]
    convex = [p for p in pieces if _reflex_count(p["polygon"]) == 0]
    if len(concave) != 1:
        return False, {}
    cp, k = concave[0]
    if k != 3 or len(convex) != 3:
        return False, {}
    return True, {"concave": cp, "triangles": convex}


def notch_solution_from(out: Path, scaled_json: Path) -> tuple[dict | None, dict]:
    """把双射匹配结果转成流水线通用的 {placed, polys} 解。"""
    f = out / "notch_ratio_match.json"
    if not f.exists():
        return None, {}
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
        scaled = json.loads(Path(scaled_json).read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    by_label = {int(p["label"]): p for p in scaled.get("pieces", [])}
    cp = by_label.get(int(doc.get("concave_piece", {}).get("label", -1)))
    if cp is None:
        return None, {}
    polys = [[[float(x), float(y)] for x, y in cp["polygon"]]]
    placed = [[int(cp["id"]), 0.0, 0.0, 0.0]]
    for t in doc.get("triangles") or []:
        p = by_label.get(int(t["label"]))
        poly = t.get("placed_polygon") or t.get("placed")
        if p is None or not poly:
            return None, {}
        polys.append([[*map(float, q)] for q in poly])
        placed.append([int(p["id"]), 0.0, 0.0, 0.0])
    return {"placed": placed, "polys": polys}, doc


def notch_ok(doc: dict, max_overlap: float = 1.0, max_mean_err: float = 0.15) -> tuple[bool, str]:
    """双射解能不能直接用：必须一一对应、贴得够准、没有明显穿透。"""
    if not doc.get("bijection_ok"):
        return False, "双射校验未通过"
    ms = doc.get("matches") or []
    if not ms:
        return False, "没有匹配结果"
    errs = [float(m.get("shape_err", 1.0)) for m in ms]
    mean_err = sum(errs) / len(errs)
    overlap = float((doc.get("coverage") or {}).get("overlap_pct", 100.0))
    gap = float((doc.get("coverage") or {}).get("gap_pct", 100.0))
    if overlap > max_overlap:
        return False, f"重叠 {overlap:.2f}% > {max_overlap}%"
    if mean_err > max_mean_err:
        return False, f"平均形状误差 {mean_err:.3f} > {max_mean_err}"
    return True, f"平均形状误差 {mean_err:.3f}，重叠 {overlap:.2f}%，缺口 {gap:.2f}%"


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
    fg_alpha_path: str | Path | None = None,
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
    # 软边 alpha 优先（01_alpha.png 是 0~255 渐变），没有才退回二值 01_mask.png
    for cand in (fg_alpha_path, fg_mask_path):
        if not cand:
            continue
        try:
            f = read_image(cand)
        except Exception:
            continue
        if f is None:
            continue
        if f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if f.shape[:2] != (h_img, w_img):
            continue
        fg = f
        break
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
        alpha_w = None
        if fg is not None:
            fg_warped = cv2.warpAffine(
                fg[y0:y1, x0:x1], mat_crop, (width, height),
                flags=cv2.INTER_LINEAR, borderValue=0,
            )
            # 软 alpha 合成：边缘按覆盖率过渡，既不出绿边也不出硬白缝
            alpha_w = np.clip(fg_warped.astype(np.float32) / 255.0, 0.0, 1.0)
            sel = sel & (alpha_w > 0.35)
        if not sel.any():
            continue
        if alpha_w is not None:
            a3 = alpha_w[sel][:, None]
            canvas[sel] = (canvas[sel].astype(np.float32) * (1 - a3)
                           + warped[sel].astype(np.float32) * a3).astype(np.uint8)
        else:
            canvas[sel] = warped[sel]
        painted += int(sel.sum())
        canvas_polys.append((pid, poly_canvas))

    if painted == 0:
        return False

    # 软边合成后拼缝处会留下发丝级白缝（二值掩膜时代的膨胀补边被去掉了）。
    # 这里把「不接触画布边缘的小白块」当作缝隙，用周围像素 inpaint 填平：
    # 既保留软边（不再有背景色包边），又不会露出白线。
    try:
        white = ((canvas > 244).all(axis=2)).astype(np.uint8)
        n_w, lab_w, st_w, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
        border_labels = set(lab_w[0, :]) | set(lab_w[-1, :]) \
            | set(lab_w[:, 0]) | set(lab_w[:, -1])
        holes = np.zeros((height, width), np.uint8)
        for i in range(1, n_w):
            if i in border_labels:
                continue
            if int(st_w[i, cv2.CC_STAT_AREA]) <= 400:
                holes[lab_w == i] = 255
        if holes.any():
            canvas = cv2.inpaint(canvas, holes, 2, cv2.INPAINT_TELEA)
            if info is not None:
                info["seam_filled_px"] = int(np.count_nonzero(holes))
    except Exception:
        pass

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
class JobCancelled(Exception):
    """用户暂停/放弃任务。流水线在下一个步骤边界立刻停手，不再往下跑。"""


def solve(
    image_path: str | Path,
    out_dir: str | Path,
    beam_time: float | None = None,
    concave: bool = True,
    log=None,
    should_cancel=None,
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
        # 步骤边界就是暂停点：每个子步骤前后都会经过 step()，所以点「暂停」后
        # 最多再等当前这一个子步骤跑完（beam 搜索可能要几十秒），不会继续往下一层走。
        if should_cancel is not None and should_cancel():
            raise JobCancelled("用户暂停")
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
            f"# 参数 beam_time={beam_time or 'auto'}s concave={concave}\n"
            f"# Python {sys.version.split()[0]}  cwd={PROJ_DIR}\n"
            + "#" * 70 + "\n"
        )

    # 0) 归一化输入：转存一份到输出目录（cv2 在 Windows 下读不了中文路径），并限制尺寸
    src_img = read_image(image_path)
    h, w = src_img.shape[:2]
    max_side = 3200
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

    # 1.2) 绿幕抠图重测：YCbCr 键控 + despill + 亚像素边界（比 ΔE 硬阈值准得多）
    keyed_img = out / "keyed.jpg"
    alpha_png = out / "key_alpha.png"
    key_info: dict = {}
    try:
        import resegment                      # 同目录模块，延迟导入避免循环依赖
        key_info = resegment.refine_pieces(
            image_path, pieces_json, pieces_json, alpha_png, keyed_img
        )
        detail = "、".join(f"{c['piece']} {c['vertices']}" for c in key_info.get("changes", []))
        step("key", f"绿幕抠图重测 {key_info.get('found')} 块：{detail or '无变化'}")
    except Exception as exc:
        key_info = {"error": str(exc)}
        step("key", f"抠图重测跳过：{exc}")

    # 1.3) 清理多边形：影子/阴影会把一条直边切成两段，合并掉近共线的假顶点
    clean = clean_pieces(pieces_json)
    if clean.get("merged"):
        detail = "、".join(f"{m['piece']} {m['from']}→{m['to']}" for m in clean["merged"])
        step("clean", f"合并共线顶点：{detail}")
    else:
        step("clean", "共线检查通过，无需合并")

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

    # 3) 先验优先：1 个凹形 + 3 个三角形 → 每个三角形只能对应一个缺口（双射）。
    #    满足且结果达标就直接采用，省掉几十秒束搜索；不满足/不达标再退回束搜索。
    sol_json = out / "beam_solutions.json"
    beam_runs: list[dict] = []
    pick = None
    n_sol = 0
    notch_sol = None
    notch_doc: dict = {}
    applicable, _prior = notch_applicable(pieces_json)
    if applicable:
        step("notch", "1 凹形 + 3 三角形：按比例双射匹配 3 个缺口…")
        _run_soft("notch_ratio_match.py",
                  ["--pieces", scaled_json, "--out", out, "--match-mode", "shape"],
                  timeout=180, logs_dir=logs)
        notch_sol, notch_doc = notch_solution_from(out, scaled_json)
        if notch_sol is None:
            step("notch", "双射匹配没产出可用结果，退回束搜索")
        else:
            ok, why = notch_ok(notch_doc)
            if ok:
                n_sol = 1
                step("notch", f"采用双射匹配（{why}）")
            else:
                step("notch", f"双射匹配不达标（{why}），退回束搜索")
                notch_sol = None
    else:
        step("notch", "碎片构成不是「1 凹形 + 3 三角形」，跳过双射匹配")

    # 4) 束搜索拼合 —— 双射已成功时跳过（时间预算自动决定，不满意自己加时重搜）
    if notch_sol is None:
        total_edges = sum(len(p.get("edges") or []) for p in piece_list)
        plan = [round(float(beam_time))] if beam_time else beam_plan(n_pieces, total_edges)
        for i, budget in enumerate(plan):
            step("beam", f"搜索拼法（第 {i + 1} 次，预算 {budget} 秒）…")
            _run("beam_search.py",
                 ["--pieces", scaled_json, "--out", out,
                  "--beam", "300", "--time", str(float(budget))],
                 timeout=budget + 240, log=step, steps=steps,
                 logs_dir=logs)
            if not sol_json.exists():
                raise RuntimeError("搜索失败：没有生成 beam_solutions.json")
            n_sol = json.loads(sol_json.read_text(encoding="utf-8")).get("n", 0)
            if not n_sol:
                continue
            pick = rerank_solutions(sol_json, names, starts)
            beam_runs.append({"budget": budget, "solutions": n_sol,
                              "len_diff": pick["diff"] if pick else None,
                              "pairs": pick["pairs"] if pick else 0})
            if beam_ok(pick, n_pieces) or time.time() - t_start > 150:
                break
        if not n_sol:
            raise RuntimeError("没有找到任何拼法，换个背景更干净的照片试试")

    # 3.5) T 型接缝补充搜索：beam 表达不了「短边顶在长边中间」（一条边被两块共用），
    #      而那正是矩形切法的常见形状，不补这一步永远看不到 A2↔B4 这种相邻关系
    split_sols: list = []
    if notch_sol is None:
        try:
            import split_search
            from assemble import Piece as _Piece      # split_search 内部也这么建
            sdata = json.loads(scaled_json.read_text(encoding="utf-8"))
            spieces = {int(d["id"]): _Piece(d) for d in sdata["pieces"]}
            found = split_search.search(spieces, time.time() + 12, cov=0.30)
            split_sols = [{"gap_pct": g, "placed": pl, "polys": [p.tolist() for p in ps]}
                          for g, pl, ps in found[:300]]
            (out / "split_solutions.json").write_text(
                json.dumps({"n": len(found), "solutions": split_sols}, ensure_ascii=False),
                encoding="utf-8")
            step("split", f"T型接缝补充 {len(found)} 个候选")
        except Exception as exc:
            step("split", f"T型接缝搜索跳过：{exc}")

        pick = rerank_solutions(sol_json, names, starts, extra_sols=split_sols)

        # 5) 并集评估
        step("eval", f"{n_sol} 个候选，评估空洞/重叠…")
        _run("eval_union.py",
             ["--solutions", sol_json, "--out", out, "--top", "4"], timeout=180, log=step,
             steps=steps, logs_dir=logs)

    # 5) 渲染成品图
    step("render", "按位姿贴回真实像素…")
    assembled = out / "assembled.jpg"
    assembled_labels = out / "assembled_labels.jpg"
    best_union = out / "best_union.json"

    # 4.5) 候选重排（beam 循环里已经算过一次，这里只做落盘）
    step("rerank", "按对接边长度一致度重排候选…")
    solution = None
    ranking: dict = {"beam_runs": beam_runs}
    if notch_sol is not None:
        solution = notch_sol
        ranking.update({
            "chosen": "notch-bijection",
            "notch_matches": [
                {"triangle": m.get("triangle_label"), "notch": m.get("notch_index"),
                 "shape_err": round(float(m.get("shape_err", 0.0)), 4)}
                for m in (notch_doc.get("matches") or [])
            ],
            "notch_coverage": notch_doc.get("coverage"),
        })
        (out / "chosen_solution.json").write_text(
            json.dumps({"ranking": ranking, **solution}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif pick:
        entry = pick["entry"]
        solution = {"placed": entry.get("placed"), "polys": entry.get("polys")}
        ranking.update({
            "chosen": f"beam#{pick['index']}",
            "mean_len_diff_pct": pick["diff"],
            "joint_pairs": pick["pairs"],
            "score": pick["score"],
            "budget_s": beam_runs[-1]["budget"] if beam_runs else None,
            "attempts": len(beam_runs),
        })
        (out / "chosen_solution.json").write_text(
            json.dumps({"ranking": ranking, **solution}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif best_union.exists():
        try:
            solution = json.loads(best_union.read_text(encoding="utf-8"))
            ranking["chosen"] = "best_union"
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
            rendered = render_assembled(
                keyed_img if keyed_img.exists() else image_path,
                scaled_json, best_union, assembled,
                names=names, labels_out=assembled_labels,
                edge_starts=starts, info=info,
                solution=solution,
                fg_mask_path=alpha_png if alpha_png.exists() else out / "01_mask.png")
        except Exception as exc:                      # 渲染失败退回排名图
            metrics["render_error"] = str(exc)
    if not rendered:
        rank = out / "41_union_rank.png"
        if rank.exists():
            write_image(assembled, read_image(rank), quality=92)
        else:
            raise RuntimeError("渲染成品图失败")

    # 6) 把「最可能的 N 种拼法」各画一张缩略图，供页面对比
    candidates: list = []
    if pick:
        for rank_i, item in enumerate(pick["top"], start=1):
            name = f"cand_{rank_i}.jpg"
            try:
                render_assembled(
                    keyed_img if keyed_img.exists() else image_path,
                    scaled_json, best_union, out / name,
                    solution={"placed": item["entry"].get("placed"),
                              "polys": item["entry"].get("polys")},
                    fg_mask_path=alpha_png if alpha_png.exists() else out / "01_mask.png",
                    max_side=520, margin=10,
                )
            except Exception:
                name = None
            m = item["metrics"]
            candidates.append({
                "rank": rank_i,
                "index": item["index"],
                "chosen": rank_i == 1,
                "file": name,
                "score": item["score"],
                "len_diff_pct": m.get("len_diff_pct"),
                "hull_gap_pct": m.get("hull_gap_pct"),
                "rect_fill_pct": m.get("rect_fill_pct"),
                "outline_vertices": m.get("outline_vertices"),
                "joint_pairs": m.get("joint_pairs"),
            })

    # 6.5) 单独列出「含 T 型接缝」的候选：短边顶在长边中间这种拼法综合分天然偏低
    #      （空隙大、长度差大），排不进前 5，但用户可能就是想看这种
    t_joint_rows = []
    for idx, entry in enumerate(split_sols):
        tinfo = solution_metrics(entry, names, starts, share=0.25)   # 别叫 info：会覆盖上面的 joints
        if tinfo is None:
            continue
        ids = [int(x[0]) for x in (entry.get("placed") or [])]
        labeled = [(names.get(pid, str(pid)), np.asarray(poly, float), starts.get(pid))
                   for pid, poly in zip(ids, entry.get("polys") or [])]
        # T 型接缝两种形态都要抓：
        #   a) 短边只贴住长边的一段（share < 90%）
        #   b) 短边整条嵌在长边里（share ≈ 100%，但两条边长度比 < 0.85）
        lens = {}
        for nm, poly_, st_ in labeled:
            for no_, a_, b_ in edges_of(poly_, st_):
                lens[f"{nm}{no_}"] = float(np.linalg.norm(np.asarray(b_) - np.asarray(a_)))
        partial = []
        for j in find_joints(labeled, share=0.25):
            la, lb = lens.get(j["a"], 0.0), lens.get(j["b"], 0.0)
            ratio = (min(la, lb) / max(la, lb)) if max(la, lb) > 0 else 1.0
            if j["share"] < 90 or ratio < 0.85:
                partial.append(j)
        if not partial:
            continue
        t_joint_rows.append({"entry": entry, "metrics": tinfo, "score": tinfo["score"],
                             "t": partial})
    t_joint_rows.sort(key=lambda r: r["score"])
    base_rank = len(candidates)
    for k, row in enumerate(t_joint_rows[:3], start=1):
        name = f"cand_{base_rank + k}.jpg"
        try:
            render_assembled(
                keyed_img if keyed_img.exists() else image_path,
                scaled_json, best_union, out / name,
                solution={"placed": row["entry"].get("placed"),
                          "polys": row["entry"].get("polys")},
                fg_mask_path=alpha_png if alpha_png.exists() else out / "01_mask.png",
                max_side=520, margin=10,
            )
        except Exception:
            name = None
        m = row["metrics"]
        candidates.append({
            "rank": base_rank + k,
            "index": -1,
            "chosen": False,
            "t_joint": True,
            "joins": [(j["a"], j["b"], j["share"]) for j in row["t"]][:3],
            "file": name,
            "score": row["score"],
            "len_diff_pct": m.get("len_diff_pct"),
            "hull_gap_pct": m.get("hull_gap_pct"),
            "rect_fill_pct": m.get("rect_fill_pct"),
            "outline_vertices": m.get("outline_vertices"),
            "joint_pairs": m.get("joint_pairs"),
        })

    steps["total"] = round(time.time() - t_start, 2)
    result = {
        "pieces": n_pieces,
        "solutions": n_sol,
        "metrics": metrics,
        "steps": steps,
        "labels": labels,
        "joints": info.get("joints", []),
        "ranking": ranking,
        "clean": clean,
        "keying": key_info,
        "candidates": candidates,
        "has_labels": bool(upload_labels),
        "files": [p.name for p in sorted(out.iterdir()) if p.is_file()],
    }
    # 7) 曲线边 + 纹理连续性分析（新增维度，只做度量与提示，不改变最终拼法）
    try:
        import curve_texture
        curve_texture.analyze(out, result)
        step("curve", "曲线边 + 纹理连续性分析完成")
    except Exception as exc:
        result["curve_texture_error"] = str(exc)

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
    ap.add_argument("--time", type=float, default=None,
                    help="束搜索时间预算（秒）。不给就按碎片数自动决定，不够吻合会自动加时")
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
