"""手撕纸条拼图：分割 → 多边形化 → 形状+纹理拼合 → 渲染效果图。

和 Voronoi 那个 demo 不同，这次是真实照片：
  背景是黑的，碎片是白色纸条，纸条上有蓝色笔迹。
  四张纸条形状都接近长方形（形状信息弱），**笔迹是唯一能对齐的纹理** ——
  正好是「形状 + 纹理」结合该发挥作用的场景。

用法：
    python 纹理拼合/tools/handmade_puzzle.py --image 手撕拼图.jpg --out 纹理拼合/手撕拼图
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon as SPoly
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))

import geometry as G  # noqa: E402
from assemble import Piece  # noqa: E402
from irregular_cut import assemble, to_pieces_json  # noqa: E402


def imread_u(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        buf.tofile(str(path))


# --------------------------------------------------------------------------
# 1) 分割：黑底 + 白纸条
# --------------------------------------------------------------------------
def segment(img: np.ndarray, th: int, min_area: float, eps_ratio: float):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (gray > th).astype(np.uint8) * 255
    # 笔迹是深蓝，可能低于阈值 -> 用外轮廓重填，把内部洞补掉
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    kept = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        cv2.fillPoly(filled, [c], 255)
        kept.append(c)
    kept.sort(key=lambda c: -cv2.contourArea(c))

    cells = []
    for c in kept:
        peri = cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps_ratio * peri, True).reshape(-1, 2).astype(float)
        if len(poly) >= 3:
            cells.append(poly)
    return cells, filled


# --------------------------------------------------------------------------
# 2) 渲染：把每块按解出的位姿贴回真实像素
# --------------------------------------------------------------------------
def render(img: np.ndarray, pieces: dict, sol: dict, margin: int = 24):
    polys = [np.asarray(p, float) for p in sol["polys"]]
    poses = [(float(it[1]), float(it[2]), float(it[3])) for it in sol["placed"]]
    ids = [int(it[0]) for it in sol["placed"]]

    pts = np.vstack(polys)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    off = mn - margin
    w = int(np.ceil(mx[0] - mn[0])) + margin * 2
    h = int(np.ceil(mx[1] - mn[1])) + margin * 2
    canvas = np.full((h, w, 3), 18, np.uint8)

    for pid, poly_w, (ang, tx, ty) in zip(ids, polys, poses):
        c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        M = np.array([[c, -s, tx - off[0]],
                      [s, c, ty - off[1]]], np.float64)
        warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderValue=(0, 0, 0))
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(poly_w - off).astype(np.int32)], 255)
        canvas[mask > 0] = warped[mask > 0]
    return canvas


def draw_labels(img: np.ndarray, cells: list[np.ndarray]) -> np.ndarray:
    out = img.copy()
    palette = [(60, 130, 245), (60, 200, 120), (230, 130, 60), (200, 90, 220),
               (60, 220, 230), (180, 70, 180)]
    for i, poly in enumerate(cells):
        p = np.round(poly).astype(np.int32)
        col = palette[i % len(palette)]
        cv2.polylines(out, [p], True, col, 3, cv2.LINE_AA)
        c = poly.mean(axis=0).astype(int)
        s = str(i + 1)
        cv2.putText(out, s, (c[0] - 16, c[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    1.7, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(out, s, (c[0] - 16, c[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    1.7, col, 3, cv2.LINE_AA)
    return out


def quality(sol: dict) -> dict:
    ps = [SPoly(p).buffer(0) for p in sol["polys"]]
    u = unary_union(ps)
    total = sum(p.area for p in ps)
    holes = sum(abs(SPoly(r).area) for r in getattr(u, "interiors", []))
    rect = u.minimum_rotated_rectangle
    return {
        "overlap_pct": round((total - u.area) / total * 100, 2),
        "holes_pct": round(holes / total * 100, 2),
        "rect_fill_pct": round(u.area / rect.area * 100, 1) if rect.area else 0.0,
    }


def seam_pairs(sol: dict):
    """解里所有真正贴在一起的相邻块对（含接缝线段与世界位姿）。"""
    from shapely.geometry import LineString
    polys = [np.asarray(p, float) for p in sol["polys"]]
    poses = [(float(it[1]), float(it[2]), float(it[3])) for it in sol["placed"]]
    out = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = SPoly(polys[i]).buffer(0).intersection(SPoly(polys[j]).buffer(0))
            if inter.is_empty:
                continue
            seg = inter if inter.geom_type == "LineString" else getattr(inter, "geoms", [None])[0]
            if seg is None or seg.geom_type != "LineString" or seg.length < 20:
                continue
            c = list(seg.coords)
            aw, bw = np.array(c[0]), np.array(c[-1])
            u = (bw - aw) / (np.linalg.norm(bw - aw) + 1e-12)
            ca = polys[i].mean(axis=0)
            sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
            out.append((aw, bw, 1.0 if sa > 0 else -1.0, poses[i], poses[j]))
    return out


def ink_continuity(sol: dict, ink: np.ndarray, k: float = 0.8) -> dict:
    """笔迹连续性：沿每条接缝采样，看两侧「是不是笔迹」是否一致。

    这张图的正确答案就是「蓝色笔迹能接上」，所以这个指标比几何指标更能说明问题。
    """
    pairs = seam_pairs(sol)
    bad = total = 0
    h, w = ink.shape

    def samp(pts):
        x = np.clip(pts[:, 0], 0, w - 1.001)
        y = np.clip(pts[:, 1], 0, h - 1.001)
        return ink[np.floor(y).astype(int), np.floor(x).astype(int)]

    def to_orig(world_pts, pose):
        ang, tx, ty = pose
        return (G.rotation(-ang) @ (world_pts - np.array([tx, ty])).T).T

    for aw, bw, side, pa, pb in pairs:
        u = (bw - aw) / (np.linalg.norm(bw - aw) + 1e-12)
        n = np.array([-u[1], u[0]]) * side
        ts = np.linspace(0.1, 0.9, 40)[:, None]
        base = aw[None, :] + (bw - aw)[None, :] * ts
        ia = samp(to_orig(base + n * k, pa))
        ib = samp(to_orig(base - n * k, pb))
        bad += int(np.sum(ia != ib))
        total += len(ia)
    return {"seams": len(pairs), "mismatch_pct": round(100.0 * bad / max(1, total), 1)}


def compose_compare(left: np.ndarray, right: np.ndarray, path: Path,
                    height: int = 620) -> None:
    """左：原图（含块编号）；右：拼合结果。做成一张对比效果图。"""
    def fit(im):
        s = height / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), height), interpolation=cv2.INTER_AREA)

    a, b = fit(left), fit(right)
    gap = 16
    canvas = np.full((height + 56, a.shape[1] + b.shape[1] + gap * 3, 3), 16, np.uint8)
    canvas[48:48 + height, gap:gap + a.shape[1]] = a
    canvas[48:48 + height, gap * 2 + a.shape[1]:gap * 2 + a.shape[1] + b.shape[1]] = b
    cv2.putText(canvas, "original  (pieces labelled)", (gap, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(canvas, "assembled  (shape + texture)", (gap * 2 + a.shape[1], 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)
    imwrite_u(path, canvas)


def main() -> int:
    ap = argparse.ArgumentParser(description="手撕纸条拼图（形状 + 纹理）")
    ap.add_argument("--image", default="手撕拼图.jpg")
    ap.add_argument("--out", default="纹理拼合/手撕拼图")
    ap.add_argument("--th", type=int, default=100, help="前景灰度阈值")
    ap.add_argument("--min-area", type=float, default=20000)
    ap.add_argument("--eps", type=float, default=0.02, help="多边形简化比例")
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--cov", type=float, default=0.35,
                    help="贴合比例下限（手撕边是波浪，用直线多边形近似后不能要求太高）")
    ap.add_argument("--overlap", type=float, default=0.02,
                    help="允许的穿透比例（占块面积；手撕边近似为直线后会轻微重叠）")
    ap.add_argument("--texture-weight", type=float, default=1.0)
    ap.add_argument("--tex-k", type=float, default=4.0,
                    help="纹理采样距离（px）。拆开的纸片边缘有暗边带，太小会被暗边带偏")
    args = ap.parse_args()

    src = imread_u(Path(args.image))
    if src is None:
        print("读不到图片：", args.image)
        return 1
    print("图像尺寸 %dx%d" % (src.shape[1], src.shape[0]))

    cells, mask = segment(src, args.th, args.min_area, args.eps)
    print("分割出 %d 块：" % len(cells), [len(c) for c in cells], "个顶点")

    data = to_pieces_json(cells)
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
    for i, p in pieces.items():
        print("  #%d 顶点=%d 面积=%.0f 边长中位=%.1f"
              % (i, p.n, p.area, float(np.median([e["length"] for e in p.edges]))))

    img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    labelled = draw_labels(src, cells)
    imwrite_u(outdir / "1_分割标注.jpg", labelled)
    imwrite_u(outdir / "0_前景掩膜.png", mask)

    # 笔迹掩膜（蓝通道明显高于红/绿）——这张图的正确答案判据
    bch, gch, rch = (src[..., i].astype(np.int16) for i in range(3))
    ink = ((bch - np.maximum(gch, rch)) > 25).astype(np.uint8)

    results = {}
    effects = {}
    for tag, tw in (("纯形状", 0.0), ("形状+纹理", args.texture_weight)):
        sol = assemble(pieces, beam=args.beam, time_budget=args.time,
                       cov=args.cov, overlap_tol=args.overlap,
                       img=(img_rgb if tw else None), tw=tw, tex_k=args.tex_k)
        if sol is None:
            print("%-9s 没有完整拼法" % tag)
            results[tag] = None
            continue
        q = quality(sol)
        ic = ink_continuity(sol, ink)
        effect = render(src, pieces, sol)
        effects[tag] = effect
        name = "2_拼合结果_纯形状.jpg" if tw == 0 else "2_拼合结果_形状加纹理.jpg"
        imwrite_u(outdir / name, effect)
        print("%-9s 代价=%.3f  空隙=%.2f%%  重叠=%.2f%%  空洞=%.2f%%  矩形贴合=%.1f%%  "
              "接缝=%d 笔迹不连续=%.1f%%  → %s"
              % (tag, sol["seam"], sol["gap_pct"], q["overlap_pct"], q["holes_pct"],
                 q["rect_fill_pct"], ic["seams"], ic["mismatch_pct"], name))
        results[tag] = {"seam": sol["seam"], "gap_pct": sol["gap_pct"], **q, **ic,
                        "file": name}
        (outdir / ("solution_纯形状.json" if tw == 0 else "solution_形状加纹理.json")).write_text(
            json.dumps({"placed": sol["placed"], "polys": sol["polys"],
                        "quality": results[tag]}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    # 对比效果图：左原图（带编号），右拼合结果
    best = effects.get("形状+纹理") if args.texture_weight else None
    if best is None:
        best = effects.get("纯形状")
    if best is not None:
        compose_compare(labelled, best, outdir / "3_对比效果图.jpg")
        print("对比效果图 → 3_对比效果图.jpg")


    (outdir / "report.json").write_text(
        json.dumps({"pieces": len(cells), "vertices": [len(c) for c in cells],
                    "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n输出目录：", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
