"""诊断：看每个碎片在不同逼近精度下的顶点与内角，确认 #2 是不是六边形。

用法: python src/inspect_vertices.py --image input/pieces2.jpg
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
    ap.add_argument("--image", default="input/pieces2.jpg")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    mask = auto_mask(bgr)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [l for l in range(1, n) if stats[l, cv2.CC_STAT_AREA] > 1500]
    print(f"碎片数 {len(comps)}")

    for idx in comps:
        comp = (labels == idx).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        print(f"\n=== 连通块 {idx}  面积 {stats[idx, cv2.CC_STAT_AREA]} ===")
        for eps in (0.008, 0.012, 0.02, 0.03):
            rough = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2).astype(float)
            refined = refine_corners(cnt, rough)
            angs = G.interior_angles(refined)
            print(f"  eps={eps:.3f}: {len(refined)} 个顶点  内角 " +
                  " ".join(f"{a:.0f}" for a in angs))
            if eps == 0.012:
                for p, a in zip(refined, angs):
                    print(f"      ({p[0]:7.1f},{p[1]:7.1f})  内角 {a:6.1f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
