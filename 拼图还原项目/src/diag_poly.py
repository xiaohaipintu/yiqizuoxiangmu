"""诊断：不同逼近精度下每个碎片的顶点数与面积偏差，挑出既简洁又贴合的表示。

用法: python src/diag_poly.py --image input/pieces4.jpg --concave
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as G  # noqa: E402
from segment import auto_mask, refine_corners  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="input/pieces4.jpg")
    ap.add_argument("--concave", action="store_true")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    mask = auto_mask(bgr)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [l for l in range(1, n) if stats[l, cv2.CC_STAT_AREA] > 1500]
    print(f"碎片数 {len(comps)}\n")

    for lbl in comps:
        comp = (labels == lbl).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        area_px = float(stats[lbl, cv2.CC_STAT_AREA])
        base = cnt if args.concave else cv2.convexHull(cnt)
        peri = cv2.arcLength(base, True)
        print(f"=== 块 {lbl}  像素面积 {area_px:.0f}  周长 {peri:.0f} ===")
        for eps in (0.006, 0.010, 0.014, 0.020, 0.028, 0.038):
            rough = cv2.approxPolyDP(base, eps * peri, True).reshape(-1, 2).astype(float)
            rough = G.drop_collinear(rough, tol_deg=6.0, px_dev=3.0)
            poly = refine_corners(cnt, rough)
            dev = (G.polygon_area(poly) - area_px) / area_px * 100
            print(f"  eps {eps:.3f}: {len(poly):2d} 顶点  面积偏差 {dev:+6.1f}%  "
                  f"边长 {[int(round(float(np.linalg.norm(poly[(i+1) % len(poly)] - poly[i])))) for i in range(len(poly))]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
