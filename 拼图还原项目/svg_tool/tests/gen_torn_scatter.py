"""手撕散落生成器：把信纸撕成带「毛边/纤维」的碎片并随机旋转平移散开。

难点（相对干净 Voronoi 切割）：边缘不再是直线，而是锯齿 + 伸出的纸纤维"毛"——
会干扰分割（轮廓毛糙、纤维可能把相邻片连起来）与拼装（没有干净的等长直边）。

用法:
  python 拼图还原项目/svg_tool/tests/gen_torn_scatter.py --n 40 --seed 7 \
      --image 拼图还原项目/svg_tool/test_out/letter_h5.png --out 拼图还原项目/svg_tool/test_out/torn_scatter
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import jigsaw_poly_solve as J


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(pathlib.Path(p).suffix or ".png", im)[1].tofile(str(p))


def ragged(poly, amp, rng):
    """把多边形每条边打成锯齿/纤维状（多频抖动 + 高频毛刺）。"""
    out = []
    m = len(poly)
    for i in range(m):
        a = poly[i]; b = poly[(i + 1) % m]
        L = np.linalg.norm(b - a)
        nb = max(3, int(L / 5))
        d = (b - a) / (L or 1.0); nx, ny = -d[1], d[0]
        for t in np.linspace(0, 1, nb, endpoint=False):
            base = a + (b - a) * t
            off = amp * rng.normal() * 0.7 + amp * 0.8 * np.sin(t * np.pi * 3.0 + i)
            out.append(base + np.array([nx, ny]) * off)
    return np.asarray(out, float)


def hairs_for(poly, rng, n=26, length=(4, 13)):
    """沿边界随机长出细纤维（短线段，端点略抖）。"""
    m = len(poly); hs = []
    for _ in range(n):
        i = rng.integers(m)
        a = poly[i]; b = poly[(i + 1) % m]
        t = rng.random(); p = a + (b - a) * t
        d = b - a; L = np.linalg.norm(d) or 1.0
        nx, ny = -d[1] / L, d[0] / L
        sgn = 1 if rng.random() > 0.5 else -1
        ln = rng.uniform(*length)
        q = p + np.array([nx, ny]) * sgn * ln + rng.normal(0, 1.5, 2)
        hs.append((p, q))
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--image", default="拼图还原项目/svg_tool/test_out/letter_h5.png")
    ap.add_argument("--out", default="拼图还原项目/svg_tool/test_out/torn_scatter")
    ap.add_argument("--amp", type=float, default=3.0, help="毛边锯齿幅度(px)")
    ap.add_argument("--hairs", type=int, default=26, help="每片纤维数")
    args = ap.parse_args()

    tex = rd(args.image)
    if tex is None:
        print("读不到纹理", args.image); return 1
    S = tex.shape[0]
    rng = np.random.default_rng(args.seed)

    cells = J.make_cells(args.n, S, args.seed)
    h, w = tex.shape[:2]

    # 每片：锯齿多边形 + 纤维 + 自带纹理的小 patch
    pieces = []
    for cell in cells:
        rag = ragged(cell, args.amp, rng)
        cen = J.poly_centroid(rag)
        hs = hairs_for(rag, rng, n=args.hairs)
        x0 = int(np.floor(rag[:, 0].min())) - 2; y0 = int(np.floor(rag[:, 1].min())) - 2
        x1 = int(np.ceil(rag[:, 0].max())) + 2; y1 = int(np.ceil(rag[:, 1].max())) + 2
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
        pw, ph = x1 - x0, y1 - y0
        rel = rag - [x0, y0]
        body = np.zeros((ph, pw), np.uint8)
        cv2.fillPoly(body, [np.round(rel).astype(np.int32)], 255)
        hair = np.zeros((ph, pw), np.uint8)
        for p0, p1 in hs:
            cv2.line(hair, tuple(np.round(p0 - [x0, y0]).astype(int)),
                     tuple(np.round(p1 - [x0, y0]).astype(int)), 255, 1)
        alpha = np.maximum(body, hair)
        color = tex[y0:y1, x0:x1].copy()
        paper = np.array([242, 249, 250], np.uint8)          # 纸色(纤维区域)
        color = np.where((body > 0)[..., None], color, paper)
        pieces.append({"rag": rag, "cen": cen, "bbox": (x0, y0), "color": color, "alpha": alpha})

    # 散开：按面积大优先，凸包(SAT/相交) 保证不重叠，留小缝（纤维会探进缝里 → 干扰）
    from shapely.geometry import Polygon
    order = sorted(range(len(pieces)), key=lambda i: -J.poly_area(pieces[i]["rag"]))
    side = int(S * 1.9)
    bg = np.full((side, side, 3), (120, 82, 60), np.uint8)   # 蓝灰背景(BGR)
    placed_hulls = []
    gt = 0
    for idx in order:
        pc = pieces[idx]
        rad = np.max(np.linalg.norm(pc["rag"] - pc["cen"], axis=1)) + 16
        for _t in range(4000):
            theta = rng.uniform(0, 2 * np.pi)
            cx = rad + rng.random() * (side - 2 * rad)
            cy = rad + rng.random() * (side - 2 * rad)
            R = J.rot(theta)
            poly_t = (pc["rag"] - pc["cen"]) @ R.T + [cx, cy]
            hull = Polygon(poly_t).convex_hull.buffer(6)      # 留 6px 缝
            if hull.bounds[0] < 0 or hull.bounds[1] < 0 or hull.bounds[2] > side or hull.bounds[3] > side:
                continue
            if any(hull.intersects(ph) for ph in placed_hulls):
                continue
            placed_hulls.append(hull)
            # 贴纹理：patch 局部坐标 -> 散落坐标
            bx, by = pc["bbox"]
            M = np.array([[R[0, 0], R[0, 1], (R @ (np.array([bx, by]) - pc["cen"]))[0] + cx],
                          [R[1, 0], R[1, 1], (R @ (np.array([bx, by]) - pc["cen"]))[1] + cy]], float)
            colw = cv2.warpAffine(pc["color"], M, (side, side), flags=cv2.INTER_LINEAR)
            alw = cv2.warpAffine(pc["alpha"], M, (side, side), flags=cv2.INTER_NEAREST)
            m3 = (alw > 0)[..., None]
            bg = np.where(m3, colw, bg)
            gt += 1
            break
        else:
            raise RuntimeError("摆放失败：画布太小")

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    wr(outdir / "torn_scatter.png", bg)
    (outdir / "gt.json").write_text(json.dumps({"pieces": gt, "n": args.n, "amp": args.amp,
                                                "hairs": args.hairs}, ensure_ascii=False), encoding="utf-8")
    print(f"手撕散落：{gt} 片（锯齿 amp={args.amp}px，每片纤维 {args.hairs} 根）→ {outdir/'torn_scatter.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
