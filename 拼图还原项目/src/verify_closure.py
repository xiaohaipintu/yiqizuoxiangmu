"""独立验证闭环拼法：直接栅格化统计 重叠面积 / 内部空洞，不依赖凸包近似。

用法: python src/verify_closure.py --json out2/30_equal_edges.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out2/30_equal_edges.json")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.json).read_text(encoding="utf-8"))
    if "closure" not in data:
        print("该结果里没有闭环记录")
        return 1

    polys = [np.array(p, float) for p in data["closure"]["poly"]]
    ids = data["closure"]["ids"]
    allp = np.vstack(polys)
    mn = allp.min(axis=0) - 5
    mx = allp.max(axis=0) + 5
    W, H = int(mx[0] - mn[0]) + 1, int(mx[1] - mn[1]) + 1
    labels = np.zeros((H, W), np.int32)
    cover = np.zeros((H, W), np.int32)   # 累计被几条多边形覆盖
    for k, p in enumerate(polys, start=1):
        pts = ((p - mn)).astype(np.int32)
        cv2.fillPoly(labels, [pts], k)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        cover += m

    each = {}
    for k, pid in enumerate(ids, start=1):
        each[pid] = int((labels == k).sum())
    union = int((labels > 0).sum())
    multi = int((cover > 1).sum())        # 两块以上同时覆盖 → 重叠
    total = sum(each.values())

    # 内部空洞：把联合区域以外、但被围住的部分
    outside = (labels == 0).astype(np.uint8)
    ff = outside.copy()
    mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    holes = int((ff == 1).sum())

    print(f"碎片 id: {ids}")
    for pid, a in each.items():
        print(f"  #{pid} 像素面积 {a}")
    print(f"联合面积 {union}   各块之和 {total}")
    print(f"重叠面积 {multi}  =  {multi / total * 100:.3f}% of sum")
    print(f"内部空洞 {holes}  =  {holes / union * 100:.3f}% of union")
    ok = multi / total < 0.004 and holes / union < 0.004
    print("结论:", "无重叠、无空洞，三块严丝合缝" if ok else "仍有缝/重叠，需要复核")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
