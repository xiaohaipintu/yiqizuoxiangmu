"""曲线边缘识别 + 纹理匹配。

背景：现在的流水线把每块碎片当成**直线多边形**（边 = 两个顶点之间的线段），
于是①真正的曲线边（弧形、波浪边、毛边）被拉成弦，长度和角度都不对；
②判断两条边是否配对只看「长度/角度」，没能利用**图像内容**——切缝两侧的画面本来
是连续的，纹理/颜色应该接得上。

本模块提供两件事：

1. `edge_metrics(poly, mask)` —— 沿多边形每条边，到实际轮廓上量「弧长 / 弦长 / 最大凹陷(bulge)」，
   得出这条边是**直边还是曲线边**、弯曲程度多大（弯曲半径 ≈ 弦长²/(8·bulge)）。

2. `edge_texture_delta(...)` —— 在**边的内侧**取一条采样带（沿内法线偏移几像素），
   得到 Lab 颜色剖面；两条边的剖面在「平移 + 翻转」后做最小平均色差，
   越小说明两侧画面越连续 → 越可能是真接缝。

两个函数都是纯几何/像素运算，不依赖 torch，只用 numpy + cv2。
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (xs - x0)[..., None]
    wy = (ys - y0)[..., None]
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return top * (1 - wy) + bot * wy


def to_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


# --------------------------------------------------------------------------
# 1) 曲线边缘：拿实际轮廓去量每条边弯了多少
# --------------------------------------------------------------------------
def _contour_of_piece(mask: np.ndarray, poly: np.ndarray):
    """从**真实掩膜**里取出这块碎片的轮廓（整数像素级，够量弯曲了）。

    注意：要量「多边形 vs 真实边界」的差，所以轮廓必须来自真实掩膜（alpha / 前景掩膜），
    不能拿多边形自己填充出来的轮廓——那样永远量不出弯曲。
    """
    region = np.zeros(mask.shape[:2], np.uint8)
    cv2.fillPoly(region, [np.round(poly).astype(np.int32)], 255)
    # 找出这块碎片所在的连通块（不能用「多边形膨胀」来裁，那样会把超出多边形的
    # 真实边界一起切掉，测出来永远是 0 —— 这正是曲线被漏掉的坑）
    nlab, labels = cv2.connectedComponents((mask > 127).astype(np.uint8))
    px = int(np.clip(round(poly[:, 0].mean()), 0, mask.shape[1] - 1))
    py = int(np.clip(round(poly[:, 1].mean()), 0, mask.shape[0] - 1))
    lab = int(labels[py, px])
    if lab == 0:      # 质心落在背景（凹形/带孔），改成"与多边形重叠最多的那个连通块"
        ys, xs = np.nonzero(region)
        vals = labels[ys, xs]
        counts = np.bincount(vals, minlength=nlab)
        counts[0] = 0
        lab = int(counts.argmax())
    if lab == 0:
        return None
    sel = (labels == lab).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def edge_metrics(poly, mask: np.ndarray, max_dev: float | None = None) -> list:
    """逐边量化「曲不曲」。

    做法：把轮廓上每个点**指派给最近的那条边**（点到线段距离），
    再在每条边内部量「路径长 / 弦长」和「相对弦的最大偏移」。

    返回 [{index, chord, arc, bulge_px, bulge_pct, arc_ratio, radius, curved}]
      chord      弦长（当前流水线用的就是它）
      arc        指派到该边的轮廓点连起来的路径长
      bulge_px   轮廓点相对弦所在直线的最大垂直偏移（像素）
      arc_ratio  arc / chord（>1.01 说明明显弯）
      radius     弯曲半径估计 chord²/(8·bulge)
      curved     判定为曲线边
    """
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    cnt = _contour_of_piece(mask, pts)
    if cnt is None:
        return [{"index": k} for k in range(n)]

    # 每个轮廓点到每条边的距离，取最近的那条（超过 max_dev 的点丢弃=角点附近）
    P = cnt
    best_d = np.full(len(P), np.inf)
    best_k = np.full(len(P), -1, dtype=np.int32)
    best_t = np.zeros(len(P))
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        d = b - a
        L2 = float(d @ d)
        if L2 < 1e-9:
            continue
        t = np.clip(((P - a) @ d) / L2, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * d[None, :]
        dist = np.linalg.norm(P - proj, axis=1)
        m = dist < best_d
        best_d[m] = dist[m]
        best_k[m] = k
        best_t[m] = t[m]

    out = []
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        d = b - a
        chord = float(np.linalg.norm(d))
        if chord < 1e-6:
            out.append({"index": k, "chord": 0.0, "arc": 0.0, "bulge_px": 0.0,
                        "bulge_pct": 0.0, "arc_ratio": 1.0, "radius": None, "curved": False})
            continue
        u = d / chord
        nrm = np.array([-u[1], u[0]])
        dev_tol = max_dev if max_dev is not None else max(6.0, 0.30 * chord)
        sel = (best_k == k) & (best_d <= dev_tol) & (best_t > 0.02) & (best_t < 0.98)
        if sel.sum() < 6:
            out.append({"index": k, "chord": round(chord, 2), "arc": round(chord, 2),
                        "bulge_px": 0.0, "bulge_pct": 0.0, "arc_ratio": 1.0,
                        "radius": None, "curved": False})
            continue
        Q = P[sel]
        off = (Q - a) @ nrm                       # 相对弦所在直线的垂直偏移
        order = np.argsort((Q - a) @ u)
        Q, off = Q[order], off[order]
        arc = float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))
        bulge = float(np.max(np.abs(off)))
        ratio = arc / chord if chord > 1e-6 else 1.0
        radius = (chord * chord) / (8.0 * bulge) if bulge > 1e-6 else None
        bulge_pct = bulge / chord * 100.0 if chord > 1e-6 else 0.0
        # 判「曲线边」要保守：凹陷得够大（≥3px），并且要么占比 ≥2%，
        # 要么弯曲半径 ≤ 1 个弦长（真的是弧）。否则量到的只是「多边形没贴合轮廓」的误差。
        curved = bool(bulge >= 3.0 and (
            bulge_pct >= 2.0
            or (radius is not None and radius <= chord)))
        out.append({
            "index": k,
            "chord": round(chord, 2),
            "arc": round(arc, 2),
            "bulge_px": round(bulge, 2),
            "bulge_pct": round(bulge / chord * 100.0, 2),
            "arc_ratio": round(ratio, 4),
            "radius": round(radius, 1) if radius else None,
            "curved": bool(curved),
        })
    return out


def curve_summary(metrics: list) -> dict:
    curved = [m for m in metrics if m.get("curved")]
    if not metrics:
        return {"edges": 0, "curved_edges": 0, "max_bulge_px": 0.0, "curved_ratio": 0.0}
    return {
        "edges": len(metrics),
        "curved_edges": len(curved),
        "curved_ratio": round(len(curved) / len(metrics), 3),
        "max_bulge_px": round(max((m.get("bulge_px") or 0.0) for m in metrics), 2),
        "max_bulge_pct": round(max((m.get("bulge_pct") or 0.0) for m in metrics), 2),
    }


# --------------------------------------------------------------------------
# 2) 纹理匹配：切缝两侧的画面连续性
# --------------------------------------------------------------------------
def edge_profile(img_lab: np.ndarray, poly, edge_index: int,
                 samples: int = 24, inset=(3.0, 9.0)) -> np.ndarray | None:
    """在一条边的**内侧**采样一条颜色带，返回 (samples, 3) 的 Lab 剖面。

    inset=(d1, d2)：沿内法线从 d1 到 d2 各采一层取平均，避开边缘过渡带。
    """
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    a, b = pts[edge_index % n], pts[(edge_index + 1) % n]
    d = b - a
    chord = float(np.linalg.norm(d))
    if chord < 4.0:
        return None
    u = d / chord
    nrm = np.array([-u[1], u[0]])
    centroid = pts.mean(axis=0)
    if float(((a + b) / 2.0 - centroid) @ nrm) > 0:      # 法线朝内
        nrm = -nrm

    ts = np.linspace(0.08, 0.92, samples)
    base = a[None, :] + d[None, :] * ts[:, None]       # (S,2)
    depths = np.linspace(inset[0], inset[1], 4)
    acc = np.zeros((samples, 3), np.float32)
    for dep in depths:
        p = base + nrm[None, :] * dep
        acc += _bilinear(img_lab, p[:, 0], p[:, 1])
    return acc / len(depths)


def profile_delta(pa: np.ndarray, pb: np.ndarray, max_shift_frac: float = 0.25) -> float:
    """两条剖面的最小平均色差（Lab ΔE），自动试「平移 + 翻转」。

    切出来的两边是同一幅画面被割开的，所以正视接缝时，两条剖面应该能对上；
    对不上（ΔE 大）说明这两条边根本不该相邻。
    """
    if pa is None or pb is None:
        return float("nan")
    S = min(len(pa), len(pb))
    pa, pb = pa[:S], pb[:S]
    best = float("inf")
    max_shift = int(S * max_shift_frac)
    for flip in (False, True):
        ref = pb[::-1] if flip else pb
        for s in range(-max_shift, max_shift + 1):
            if s >= 0:
                x, y = pa[s:], ref[: S - s]
            else:
                x, y = pa[: S + s], ref[-s:]
            if len(x) < max(6, S // 3):
                continue
            d = float(np.mean(np.linalg.norm(x - y, axis=1)))
            if d < best:
                best = d
    return round(best, 2) if np.isfinite(best) else float("nan")


def piece_edges_curved(poly, mask):
    """便捷封装：返回 (每条边的度量, 汇总)。"""
    m = edge_metrics(poly, mask)
    return m, curve_summary(m)


# --------------------------------------------------------------------------
# 3) 接进流水线：读任务目录，把「曲线边」和「纹理连续性」写回 result
# --------------------------------------------------------------------------
def analyze(out_dir, result: dict, image_name: str = "keyed.jpg",
            alpha_name: str = "key_alpha.png") -> dict:
    """给一次任务补上曲线/纹理分析，原地更新 result。

    产出：
      result["curves"]            {碎片名: {summary, edges:[…]}}
      result["joints"][i]["texture_dE"]   每条接缝两侧的画面色差（越小越连续）
      result["texture"]           {edge_pairs, mean_dE, best, worst}
    """
    import json
    from pathlib import Path

    import pipeline as P                       # 复用命名/边序，保证和 A1/A2 对得上

    out = Path(out_dir)
    pieces_json = out / "pieces.json"
    if not pieces_json.exists():
        return result
    pieces = json.loads(pieces_json.read_text(encoding="utf-8")).get("pieces") or []

    alpha_path = out / alpha_name
    mask = None
    if alpha_path.exists():
        a = P.read_image(alpha_path)
        mask = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    img_path = out / image_name
    lab = to_lab(P.read_image(img_path)) if img_path.exists() else None

    POLY, START, NAME, PROF = {}, {}, {}, {}
    curves = {}
    for i, piece in enumerate(pieces):
        pid = int(piece["id"])
        name = P.piece_name(i)
        poly = np.asarray(piece["polygon"], dtype=np.float64)
        NAME[pid], POLY[pid] = name, poly
        START[pid] = P.edge_start_index(poly)
        if mask is not None:
            m, summ = piece_edges_curved(poly, mask)
            # 把度量绑到「边名」上（A1/A2…），和 joints 里的名字一致
            by_name = {}
            for no, a, b in P.edges_of(poly, START[pid]):
                met = next((x for x in m if x.get("index") == no - 1), {})
                by_name[f"{name}{no}"] = met
            curves[name] = {"summary": summ, "edges": by_name}
        if lab is not None:
            for no, a, b in P.edges_of(poly, START[pid]):
                PROF[f"{name}{no}"] = edge_profile(lab, poly, no - 1)

    result["curves"] = curves

    # 纹理连续性：逐条接缝比两侧剖面
    tex = []
    for j in (result.get("joints") or []):
        pa, pb = PROF.get(j["a"]), PROF.get(j["b"])
        d = profile_delta(pa, pb) if (pa is not None and pb is not None) else float("nan")
        j["texture_dE"] = None if not np.isfinite(d) else d
        tex.append({"a": j["a"], "b": j["b"], "dE": j["texture_dE"]})
    ds = [t["dE"] for t in tex if t["dE"] is not None]
    result["texture"] = {
        "edge_pairs": len(tex),
        "mean_dE": round(float(np.mean(ds)), 2) if ds else None,
        "best": min(tex, key=lambda t: t["dE"]) if ds else None,
        "worst": max(tex, key=lambda t: t["dE"]) if ds else None,
        "note": "ΔE = 接缝两侧 3~9px 内侧采样带的平均色差；越小说明画面越连续、越可能真的相邻",
    }
    # 全局提示：所有接缝都很大，说明纹理这条线索不可用（比如纯色纸片）
    if len(ds) >= 2:
        lo, hi = min(ds), max(ds)
        if hi < lo * 1.5 or lo > 25:
            result["texture"]["warning"] = (
                "各接缝 ΔE 差别不大（%0.1f~%0.1f），纹理判据区分度低——" % (lo, hi)
                + "多半是纯色纸片，请以几何判据为准")
    return result

# --------------------------------------------------------------------------
# 自测用：造一张带「曲线边 + 纹理」的图，验证识别确实能看出来
# --------------------------------------------------------------------------
def _selftest(tmpdir: str) -> int:
    import os
    import sys
    h = w = 420
    img = np.full((h, w, 3), (40, 180, 40), np.uint8)          # 绿幕
    # 一块「有曲线边」的碎片：左边是波浪、右边是圆弧
    poly = np.array([[80, 120], [220, 100], [330, 180], [300, 320], [110, 300]], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    # 纹理：在碎片内部画斜条纹（让纹理匹配有东西可比）
    yy, xx = np.mgrid[0:h, 0:w]
    stripes = (((xx + yy) // 12) % 2) == 0
    img[mask > 0] = np.where(stripes[mask > 0][:, None],
                             np.array([200, 60, 200]), np.array([120, 30, 120])).astype(np.uint8)

    sub = np.asarray(poly, np.float64)
    m, summ = piece_edges_curved(sub, mask)
    print("曲线识别：", summ)
    ok = summ["edges"] == 5
    # 直边（这张图的边都是直线，bulge 应该很小）
    ok &= summ["max_bulge_px"] < 2.0
    print("  直边样本 bulge =", summ["max_bulge_px"], "px")

    # 再来一个真·曲线：掩膜画圆，但用「8 边形」当多边形 → 每条边跨过一段圆弧，bulge 应该很大
    mask2 = np.zeros((h, w), np.uint8)
    cv2.circle(mask2, (200, 200), 120, 255, -1)
    octagon = np.array([[200 + 120 * math.cos(k * math.pi / 4),
                         200 + 120 * math.sin(k * math.pi / 4)] for k in range(8)])
    m2, summ2 = piece_edges_curved(octagon, mask2)
    print("曲线样本：", summ2)
    ok &= summ2["max_bulge_px"] > 6.0        # 8 边形 vs 圆：单边凹陷 ≈ R(1-cos22.5°) ≈ 9px
    ok &= summ2["curved_edges"] == 8

    # 纹理剖面：同一块的两条不同边，ΔE 应该 > 自己和自己
    lab = to_lab(img)
    p_same = edge_profile(lab, sub, 2)
    p_other = edge_profile(lab, sub, 4)
    d_same = profile_delta(p_same, p_same)
    d_diff = profile_delta(p_same, p_other)
    print(f"  纹理：自身 ΔE={d_same:.2f}  另一条边 ΔE={d_diff:.2f}")
    ok &= (d_same < 0.01) and (d_diff >= 0)

    print("自测：", "通过 ✅" if ok else "失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftest(sys.argv[1] if len(sys.argv) > 1 else "."))
