"""第一步：从绿色背景的照片里把红色纸片分割出来，拟合成干净的多边形。

用法:
    python src/segment.py --image input/pieces.jpg --out out
输出:
    out/pieces.json      每块碎片的顶点、面积、边长、内角
    out/01_mask.png      前景掩膜（已剔除小于 --min-area 的杂点，背景干净）
    out/01_mask_raw.png  未清理的原始掩膜，仅排查用
    out/01_alpha.png     软边蒙版（0~255 渐变），贴图用它才不会带背景色包边
    out/02_pieces.png    编号 + 边编号的调试图
    out/03_cutouts.png   逐块抠图预览（白底网格），用来肉眼检查抠得干不干净
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as G  # noqa: E402


def red_mask(bgr: np.ndarray) -> np.ndarray:
    """红色纸张 vs 绿色背景：用 HSV 色相分离，再形态学去噪。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (((h <= 12) | (h >= 168)) & (s >= 90) & (v >= 40)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask


def background_delta(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从画面四边取背景色，返回 (每个像素与背景的 Lab 色差 ΔE, 背景色本身)。"""
    h, w = bgr.shape[:2]
    b = max(4, min(h, w) // 40)
    border = np.vstack(
        [
            bgr[:b].reshape(-1, 3),
            bgr[-b:].reshape(-1, 3),
            bgr[:, :b].reshape(-1, 3),
            bgr[:, -b:].reshape(-1, 3),
        ]
    )
    bg = np.median(border, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return np.linalg.norm(lab - bg_lab, axis=2), bg


def auto_mask(bgr: np.ndarray, delta_e: float = 40.0) -> np.ndarray:
    """通用做法：从画面四边取背景色，凡与背景色差超过阈值的是前景。

    这样换一张图（红纸、紫纸、任意底色）都不用改阈值；纸上印的深色小标记也归入前景，
    不会在碎片内部挖洞。
    """
    d, _bg = background_delta(bgr)
    mask = (d > delta_e).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask


def drop_specks(mask: np.ndarray, min_area: float) -> tuple[np.ndarray, int, int]:
    """抹掉面积 < min_area 的连通块。

    这些是背景上的小污点/纸屑，本来就不是碎片（后续也不会被当成 pieces），
    但会实实在在地画在 01_mask.png 上，看起来就是“抠图没抠干净”。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    dropped = 0
    dropped_area = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == i] = 255
        else:
            dropped += 1
            dropped_area += area
    return keep, dropped, dropped_area


def soft_alpha(mask: np.ndarray, delta: np.ndarray | None = None,
               lo: float = 35.0, hi: float = 55.0) -> np.ndarray:
    """软边蒙版：边缘按“前景程度”给 0~255 的渐变，而不是非黑即白。

    二值掩膜 + 膨胀贴图会在碎片外围带一条背景色包边（实测成品每块都有绿边）；
    这里让抗锯齿/半背景像素带上部分透明度，贴上去就和底色自然融合。
    掩膜外 1px 内允许渐变，再远一律透明，避免把阴影当成半透明前景渗出去。
    """
    if delta is not None:
        grad = np.clip((delta - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    else:
        grad = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0
    allowed = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    grad = np.where(allowed, grad, 0.0)
    return (np.clip(grad, 0.0, 1.0) * 255).astype(np.uint8)


def render_cutouts(bgr: np.ndarray, alpha: np.ndarray, pieces: list,
                   path: pathlib.Path, tile: int = 260, pad: int = 14) -> None:
    """把每块碎片按 alpha 抠出来，排在白底网格里 —— 抠得干不干净一眼就看得到。"""
    if not pieces:
        return
    h_img, w_img = bgr.shape[:2]
    tiles = []
    for p in pieces:
        poly = np.array(p["polygon"], np.float64)
        x0 = max(int(np.floor(poly[:, 0].min())) - 2, 0)
        y0 = max(int(np.floor(poly[:, 1].min())) - 2, 0)
        x1 = min(int(np.ceil(poly[:, 0].max())) + 2, w_img)
        y1 = min(int(np.ceil(poly[:, 1].max())) + 2, h_img)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = bgr[y0:y1, x0:x1].astype(np.float32)
        a = (alpha[y0:y1, x0:x1].astype(np.float32) / 255.0)[..., None]
        comp = (crop * a + 255.0 * (1.0 - a)).astype(np.uint8)   # 白底合成
        h, w = comp.shape[:2]
        s = min((tile - 2 * pad) / max(w, 1), (tile - 2 * pad) / max(h, 1), 1.0)
        comp = cv2.resize(comp, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA)
        tiles.append((p["id"], comp))

    cols = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
    for k, (pid, comp) in enumerate(tiles):
        r, c = divmod(k, cols)
        cy, cx = r * tile, c * tile
        oy = cy + (tile - comp.shape[0]) // 2
        ox = cx + (tile - comp.shape[1]) // 2
        canvas[oy:oy + comp.shape[0], ox:ox + comp.shape[1]] = comp
        cv2.rectangle(canvas, (cx, cy), (cx + tile - 1, cy + tile - 1), (222, 222, 222), 1)
        cv2.putText(canvas, f"#{pid}", (cx + 8, cy + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), canvas)


def residue_stats(mask: np.ndarray, pieces: list) -> list:
    """每块碎片的多边形内，有多少像素其实不是前景（= 拟合时被多边形“撑”进来的背景）。"""
    rows = []
    for p in pieces:
        poly = np.array(p["polygon"], np.int32)
        canvas = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(canvas, [poly], 255)
        inside = canvas > 0
        total = int(np.count_nonzero(inside))
        leak = int(np.count_nonzero(inside & (mask == 0)))
        rows.append((p["id"], total, leak, 100.0 * leak / max(1, total)))
    return rows


def _line_from_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘直线：返回 (直线上一点, 单位方向)。"""
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy], dtype=float)
    d /= np.linalg.norm(d) + 1e-12
    return np.array([x0, y0], dtype=float), d


def _intersect(p1, d1, p2, d2) -> np.ndarray | None:
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=float)
    det = np.linalg.det(a)
    if abs(det) < 1e-9:
        return None
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    t = np.linalg.solve(a, b)[0]
    return p1 + t * d1


def refine_corners(contour: np.ndarray, rough: np.ndarray, band: float = 9.0, max_shift: float = 45.0) -> np.ndarray:
    """用轮廓点分别拟合每条边所在直线，再求相邻直线交点作为精确角点。

    拍照得到的纸片角有圆角/毛边，approxPolyDP 给的角点会偏内；拟合直线求交可以显著提精度。
    """
    pts = contour.reshape(-1, 2).astype(float)
    lines = []
    n = len(rough)
    for i in range(n):
        a = rough[i]
        b = rough[(i + 1) % n]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-6:
            return rough
        u = ab / L
        normal = np.array([-u[1], u[0]])
        # 只保留投影落在这条边上、且离直线足够近的轮廓点
        rel = pts - a
        proj = rel @ u
        dist = np.abs(rel @ normal)
        sel = (proj > -2.0) & (proj < L + 2.0) & (dist < band)
        seg = pts[sel]
        if len(seg) < 8:
            lines.append((a, u))
        else:
            lines.append(_line_from_points(seg))
    out = []
    for i in range(n):
        p1, d1 = lines[i]
        p2, d2 = lines[(i + 1) % n]
        c = _intersect(p1, d1, p2, d2)
        if c is None or np.linalg.norm(c - rough[i]) > max_shift:
            c = rough[i]
        out.append(c)
    return np.array(out, dtype=float)


def edge_fits(cnt: np.ndarray, poly: np.ndarray, band: float = 8.0) -> list[dict]:
    """每条边上参与拟合的轮廓点数量与最大偏差，用于判断该边可信度。"""
    pts = cnt.reshape(-1, 2).astype(float)
    fits = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-6:
            fits.append({"index": i, "n": 0, "rms": 99.0})
            continue
        u = ab / L
        nrm = np.array([-u[1], u[0]])
        rel = pts - a
        proj, dist = rel @ u, np.abs(rel @ nrm)
        sel = (proj > 1.0) & (proj < L - 1.0) & (dist < band)
        d = dist[sel]
        fits.append(
            {"index": i, "n": int(sel.sum()), "rms": float(np.sqrt((d**2).mean())) if len(d) else 99.0}
        )
    return fits


def polygon_of_component(comp_mask: np.ndarray, pixel_area: float,
                         allow_concave: bool = False) -> tuple[np.ndarray, list[dict]] | None:
    cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    # 默认用凸包压掉毛边凹口；碎片本身可能是凹多边形（凹槽要被别的碎片填），此时直接沿轮廓拟合
    base = cnt if allow_concave else cv2.convexHull(cnt)
    peri = cv2.arcLength(base, True)
    # 逼近精度取 1.2%：太大（2%）会把接近直线的真拐角并掉（六边形被吃成五边形）
    rough = cv2.approxPolyDP(base, 0.012 * peri, True).reshape(-1, 2).astype(float)
    rough = G.drop_collinear(rough, tol_deg=6.0, px_dev=3.0)
    if len(rough) < 3:
        return None

    poly = refine_corners(cnt, rough)
    # 每个精修角点必须贴着真实轮廓，否则退回粗多边形；防止圆角/阴影把边外推过头
    pts = cnt.reshape(-1, 2).astype(float)
    d = np.linalg.norm(pts[:, None, :] - poly[None, :, :], axis=2).min(axis=0)
    if (d > 14.0).any():
        poly = rough.copy()
    poly = G.drop_collinear(poly, tol_deg=6.0, px_dev=3.0)

    # 多边形面积必须和真实像素面积吻合，否则用粗多边形
    if abs(G.polygon_area(poly) - pixel_area) / max(pixel_area, 1.0) > 0.06:
        poly = G.drop_collinear(rough.copy(), tol_deg=6.0, px_dev=3.0)
    return G.ensure_ccw(poly), edge_fits(cnt, poly)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="input/pieces.jpg")
    ap.add_argument("--out", default="out")
    ap.add_argument("--min-area", type=float, default=1500.0)
    ap.add_argument("--mode", choices=("auto", "red"), default="auto")
    ap.add_argument("--delta-e", type=float, default=40.0)
    ap.add_argument("--concave", action="store_true",
                    help="碎片本身是凹多边形（有凹槽），沿真实轮廓拟合而不是凸包")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"读不到图片: {args.image}")
        return 1
    print(f"图像尺寸: {bgr.shape[1]}x{bgr.shape[0]}")

    mask = red_mask(bgr) if args.mode == "red" else auto_mask(bgr, args.delta_e)
    delta = None
    if args.mode == "auto":
        delta, bg = background_delta(bgr)
        print("背景色 BGR %s" % bg.tolist())
    cv2.imwrite(str(outdir / "01_mask_raw.png"), mask)
    # 剔掉背景上的小污点/纸屑：它们本就不算碎片，但会画在掩膜上，看起来就是“没抠干净”
    mask, n_specks, speck_area = drop_specks(mask, args.min_area)
    alpha = soft_alpha(mask, delta)
    cv2.imwrite(str(outdir / "01_mask.png"), mask)
    cv2.imwrite(str(outdir / "01_alpha.png"), alpha)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pieces = []
    for lbl in range(1, n):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < args.min_area:
            continue
        comp = (labels == lbl).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        res = polygon_of_component(comp, area, allow_concave=args.concave)
        if res is None:
            continue
        poly, fits = res
        pieces.append(
            {
                "label": lbl,
                "pixel_area": area,
                "polygon": poly.tolist(),
                "area": G.polygon_area(poly),
                "centroid": G.centroid(poly).tolist(),
                "edge_fits": fits,
            }
        )

    # 编号：从上到下、从左到右
    pieces.sort(key=lambda p: (round(p["centroid"][1] / 120.0), p["centroid"][0]))
    for i, p in enumerate(pieces, start=1):
        p["id"] = i
        p["perimeter"] = G.perimeter(np.array(p["polygon"]))
        p["edges"] = G.edges(np.array(p["polygon"]))
        p["interior_angles"] = G.interior_angles(np.array(p["polygon"]))
        p["bbox"] = G.bbox(np.array(p["polygon"]))

    # 调试渲染
    vis = bgr.copy()
    palette = [
        (60, 60, 255), (60, 200, 255), (0, 200, 0), (255, 120, 0),
        (255, 0, 255), (0, 255, 255), (200, 0, 200), (0, 128, 255),
    ]
    for p in pieces:
        poly = np.array(p["polygon"], dtype=np.int32)
        col = palette[(p["id"] - 1) % len(palette)]
        cv2.polylines(vis, [poly], True, col, 2)
        c = tuple(int(v) for v in p["centroid"])
        cv2.putText(vis, f"#{p['id']}", (c[0] - 18, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(vis, f"#{p['id']}", (c[0] - 18, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
        for e in p["edges"]:
            m = (int(e["mid"][0]), int(e["mid"][1]))
            cv2.putText(vis, f"{p['id']}e{e['index']}", m, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            cv2.putText(vis, f"{p['id']}e{e['index']}", m, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(outdir / "02_pieces.png"), vis)
    render_cutouts(bgr, alpha, pieces, outdir / "03_cutouts.png")

    with open(outdir / "pieces.json", "w", encoding="utf-8") as f:
        json.dump({"image": args.image, "pieces": pieces}, f, ensure_ascii=False, indent=2)

    print(f"共 {len(pieces)} 块碎片")
    print(f"{'id':>3} {'顶点':>4} {'面积':>9} {'面积偏差':>8} {'周长':>7} {'边序号:长度(px)':<38} 内角(°)")
    for p in pieces:
        lens = " ".join(f"{e['index']}:{e['length']:.0f}" for e in p["edges"])
        angs = " ".join(f"{a:.0f}" for a in p["interior_angles"])
        dev = (p["area"] - p["pixel_area"]) / p["pixel_area"] * 100
        print(f"{p['id']:>3} {len(p['polygon']):>4} {p['area']:>9.0f} {dev:>7.1f}% {p['perimeter']:>7.0f} {lens:<38} {angs}")
    total = sum(p["area"] for p in pieces)
    print(f"碎片总面积 {total:.0f} px², 外接方形边长约 {total ** 0.5:.1f} px")
    print(f"掩膜清理：剔除杂点 {n_specks} 个（共 {speck_area:.0f} px）")
    print("残留自检（多边形内夹带的背景像素）：")
    for pid, tot, leak, pct in residue_stats(mask, pieces):
        print(f"  #{pid}  多边形内 {tot} px  夹带背景 {leak} px ({pct:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
