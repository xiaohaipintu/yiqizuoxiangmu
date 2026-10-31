"""绿幕抠图（chroma key）：YCbCr 色度键控 + despill + 软 alpha + 亚像素边界。

为什么不用 segment.py 的「ΔE > 阈值」：
纸片边缘有 3~5px 的模糊过渡带，硬阈值切在哪一侧全看运气 —— 实测边长被放大 3%~9%，
影子会被算成前景（C 多出一个 170° 的假顶点），贴图时还会把一圈绿背景带进成品图。

这里按影视绿幕的流程做：
1. **色度键控**：转到 YCbCr，只看 Cr/Cb（颜色），不看 Y（亮度）→ 影子压暗背景但色度不变，天然排除；
2. **软 alpha**：alpha = clamp((d - lo) / (hi - lo))，0=背景 1=纸片，边缘是渐变而非二值；
3. **despill**：把边缘残留的绿色压到 max(R,B) 以下，去掉"绿边"；
4. **亚像素边界**：沿每条边的法线采样 alpha，找 0.5 交点 → 直线拟合 → 相邻直线求交得角点，
   精度约 0.1px，不再受像素栅格限制。

输出：
    pieces.json       多边形被替换成精确值（面积/质心/边表同步重算）
    key_alpha.png     软 alpha 蒙版（贴图合成时用，边界干净）
    keyed.jpg         despill 之后的整图
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402

ALPHA_LO = 4.0        # 色度差小于它：纯背景
ALPHA_HI = 26.0       # 色度差大于它：纯纸片
MIN_AREA = 1200.0     # 连通块最小面积
SCAN_HALF = 7.0       # 沿法线扫描 ±7px
SCAN_STEP = 0.35


# --------------------------------------------------------------------------
# 1) 绿幕键控 → 软 alpha
# --------------------------------------------------------------------------
def key_matte(img: np.ndarray):
    """色度键控：alpha = 该像素色度差 / 纸片与背景的色度差 = 颜色混合比例。

    alpha=0.5 恰好对应「纸片占半个像素」，也就是真实边界 —— 这是把边长抠准的关键。
    """
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    h, w = ycc.shape[:2]
    band = max(4, min(h, w) // 200)
    border = np.concatenate([
        ycc[:band].reshape(-1, 3), ycc[-band:].reshape(-1, 3),
        ycc[:, :band].reshape(-1, 3), ycc[:, -band:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)                        # (Y, Cr, Cb)
    d = np.sqrt((ycc[:, :, 1] - bg[1]) ** 2 + (ycc[:, :, 2] - bg[2]) ** 2)
    # 纸片本色：色度差最大的那一小撮像素的中位数（≈ 纯纸片）
    if d.max() <= 1e-6:
        return np.zeros((h, w), np.float32), bg, 1.0
    thr = np.percentile(d, 99.5) * 0.75
    core = d > thr
    d_piece = float(np.median(d[core])) if core.sum() > 100 else float(d.max())
    alpha = np.clip(d / max(d_piece, 1e-6), 0.0, 1.0).astype(np.float32)
    return alpha, bg, d_piece


def despill(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """去掉边缘残留的绿：G 不超过 max(R,B)。内部（alpha≈1）基本不受影响。"""
    out = img.astype(np.float32)
    b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    limit = np.maximum(r, b)
    excess = np.clip(g - limit, 0, None) * (1.0 - alpha)       # 边缘才压
    out[:, :, 1] = g - excess
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 2) 亚像素边界：沿法线找 alpha=0.5
# --------------------------------------------------------------------------
def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = xs - x0
    wy = ys - y0
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return top * (1 - wy) + bot * wy


def _edge_points(alpha: np.ndarray, a: np.ndarray, b: np.ndarray, centroid: np.ndarray):
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 12:
        return None
    u = d / L
    n = np.array([-u[1], u[0]], dtype=np.float64)
    if float(((a + b) / 2 - centroid) @ n) < 0:
        n = -n                                        # 法线朝外
    ts = np.linspace(0.1, 0.9, 32)
    offs = np.arange(-SCAN_HALF, SCAN_HALF + 1e-9, SCAN_STEP)
    base = a[None, :] + d[None, :] * ts[:, None]
    pts = base[:, None, :] + n[None, None, :] * offs[None, :, None]
    prof = _bilinear(alpha, pts[..., 0], pts[..., 1])          # (T,S) 0=外 1=内
    found = []
    for ti in range(prof.shape[0]):
        row = prof[ti]
        for si in range(len(offs) - 1, 0, -1):                # 由外向内找 0.5 交点
            c_out, c_in = row[si], row[si - 1]
            if c_in >= 0.5 > c_out:
                span = c_in - c_out
                frac = 0.0 if abs(span) < 1e-9 else (0.5 - c_out) / span
                s = offs[si - 1] + frac * (offs[si] - offs[si - 1])
                found.append(base[ti] + n * s)
                break
    if len(found) < 8:
        return None
    return np.asarray(found, dtype=np.float64)


def _fit_line(pts: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return float(vx), float(vy), float(x0), float(y0)


def _intersect(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    den = vx1 * vy2 - vy1 * vx2
    if abs(den) < 1e-9:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / den
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float64)


def refine_contour(alpha: np.ndarray, cnt: np.ndarray, eps_ratio: float = 0.008) -> np.ndarray:
    """粗轮廓 → 逐边亚像素拟合 → 相邻直线求交，得到精确多边形。

    eps_ratio 保持 0.008：试过 0.004（顶点数更贴近真值 5/6/4），但亚像素噪声会冒出
    6~27px 的毛刺尖刺，拼合反而变差（对接边长差 0.93% → 1.76%，碎片之间甚至不相接）。
    宁可略粗、互相抵消，也不要毛刺。
    """
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True).reshape(-1, 2).astype(np.float64)
    if len(approx) < 3:
        return approx
    centroid = approx.mean(axis=0)
    lines = []
    for k in range(len(approx)):
        pts = _edge_points(alpha, approx[k], approx[(k + 1) % len(approx)], centroid)
        lines.append(_fit_line(pts) if pts is not None else None)
    out = []
    for k in range(len(approx)):
        l1, l2 = lines[k - 1], lines[k]
        p = _intersect(l1, l2) if (l1 and l2) else None
        out.append(p if p is not None else approx[k])
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def refine_pieces(image_path: Path, pieces_json: Path, out_json: Path,
                  alpha_out: Path | None = None, keyed_out: Path | None = None,
                  tol_angle: float = 8.0) -> dict:
    img = P.read_image(image_path)
    alpha, _bg, _dp = key_matte(img)
    hard = (alpha >= 0.5).astype(np.uint8) * 255
    hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) > MIN_AREA]

    doc = json.loads(Path(pieces_json).read_text(encoding="utf-8"))
    pieces = doc.get("pieces", [])
    report = {"found": len(cnts), "pieces": len(pieces), "changes": []}

    used = set()
    for i, piece in enumerate(pieces):
        cx, cy = piece["centroid"]
        best, best_d = None, 1e18
        for j, c in enumerate(cnts):
            if j in used:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            dist = math.hypot(M["m10"] / M["m00"] - cx, M["m01"] / M["m00"] - cy)
            if dist < best_d:
                best_d, best = dist, j
        if best is None:
            continue
        diag = math.hypot(piece["bbox"][2] - piece["bbox"][0], piece["bbox"][3] - piece["bbox"][1])
        if best_d > 0.25 * max(diag, 1.0):
            continue
        used.add(best)
        poly = refine_contour(alpha, cnts[best])
        poly = P.simplify_polygon(poly, tol_angle)
        if len(poly) < 3:
            continue
        old_len = sorted(e["length"] for e in piece["edges"])
        doc["pieces"][i] = P._rebuild_piece(piece, np.asarray(poly))
        new_len = sorted(e["length"] for e in doc["pieces"][i]["edges"])
        report["changes"].append({
            "piece": P.piece_name(i),
            "vertices": f"{len(old_len)}→{len(new_len)}",
            "edges_old": [round(x, 1) for x in old_len],
            "edges_new": [round(x, 1) for x in new_len],
            "area_old": round(float(piece["area"]), 1),
            "area_new": round(float(doc["pieces"][i]["area"]), 1),
        })

    Path(out_json).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    if alpha_out:
        P.write_image(alpha_out, (alpha * 255).astype(np.uint8))
    if keyed_out:
        P.write_image(keyed_out, despill(img, alpha), quality=92)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="绿幕抠图（chroma key）重测碎片")
    ap.add_argument("--image", required=True)
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha-out", default=None)
    ap.add_argument("--keyed-out", default=None)
    ap.add_argument("--tol-angle", type=float, default=8.0)
    args = ap.parse_args()
    rep = refine_pieces(Path(args.image), Path(args.pieces), Path(args.out),
                        Path(args.alpha_out) if args.alpha_out else None,
                        Path(args.keyed_out) if args.keyed_out else None,
                        args.tol_angle)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
