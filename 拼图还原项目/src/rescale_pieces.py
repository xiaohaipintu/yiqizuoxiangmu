"""按「像素面积 / 多边形面积」给每块做相似缩放补偿，输出 pieces_scaled.json。

薄小碎片的角被圆角+毛边吃掉，拟合出的多边形会系统性偏小（面积差 5~20%），
用这一步把多边形放大到与像素面积一致的尺度，再做拼接。

用法: python src/rescale_pieces.py --pieces out4/pieces.json --out out4/pieces_scaled.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out4/pieces.json")
    ap.add_argument("--out", default="out4/pieces_scaled.json")
    ap.add_argument("--max-scale", type=float, default=1.15)
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    out = copy.deepcopy(data)
    for d, o in zip(data["pieces"], out["pieces"]):
        poly = np.array(d["polygon"], float)
        area = G.polygon_area(poly)
        s = math.sqrt(float(d["pixel_area"]) / area)
        s = min(s, args.max_scale)
        c = poly.mean(axis=0)
        poly = (poly - c) * s + c
        o["polygon"] = poly.tolist()
        o["area"] = G.polygon_area(poly)
        o["centroid"] = poly.mean(axis=0).tolist()
        o["edges"] = G.edges(poly)
        print(f"#{d['id']}: 缩放 ×{s:.3f}  面积 {area:.0f} → {o['area']:.0f} (像素 {d['pixel_area']:.0f})")
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
