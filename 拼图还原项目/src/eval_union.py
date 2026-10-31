"""用 shapely 真实并集评估拼法：内部空洞面积 / 重叠面积（凹形拼合的正确度量）。

用法: python src/eval_union.py --solutions out4/beam_solutions.json --out out4 --top 6
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from shapely.geometry import Polygon as SPoly
from shapely.ops import unary_union

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assemble import draw  # noqa: E402
import cv2  # noqa: E402


def evaluate(polys):
    ps = [SPoly(np.asarray(p, float)).buffer(0) for p in polys]
    u = unary_union(ps)
    s = sum(p.area for p in ps)
    holes = sum(abs(SPoly(r).area) for r in getattr(u, "interiors", []))
    overlap = s - u.area
    return {"holes_pct": holes / s * 100.0, "overlap_pct": overlap / s * 100.0,
            "union_area": u.area, "sum_area": s}


def hull_gap(polys):
    pts = np.vstack(polys).astype(np.float32)
    hull = abs(float(cv2.contourArea(cv2.convexHull(pts))))
    s = sum(SPoly(np.asarray(p, float)).buffer(0).area for p in polys)
    return max(0.0, hull - s) / s * 100.0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", default="out4/beam_solutions.json")
    ap.add_argument("--out", default="out4")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.solutions).read_text(encoding="utf-8"))
    sols = data["solutions"]
    print(f"候选 {len(sols)} 个")
    ranked = []
    for sol in sols:
        m = evaluate(sol["polys"])
        m["gap_pct"] = hull_gap(sol["polys"])
        ranked.append((m["gap_pct"], m, sol))
    ranked.sort(key=lambda x: x[0])
    for i, (score, m, sol) in enumerate(ranked[:8], 1):
        print(f"  {i}: 外露缝 {m['gap_pct']:5.2f}%   重叠 {m['overlap_pct']:5.2f}%   "
              f"顺序 {[r[0] for r in sol['placed']]}")

    tiles = []
    for i, (score, m, sol) in enumerate(ranked[: args.top], 1):
        polys = [np.array(p, float) for p in sol["polys"]]
        ids = [r[0] for r in sol["placed"]]
        im = draw(polys, ids, size=430, margin=12)
        h, w = im.shape[:2]
        cv = np.full((h + 80, max(w + 20, 450), 3), 255, np.uint8)
        cv[36:36 + h, 10:10 + w] = im
        cv2.putText(cv, f"#{i}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(cv, "outer slivers %.2f%%  overlap %.2f%%" % (m["gap_pct"], m["overlap_pct"]),
                    (10, 36 + h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 70, 160), 1, cv2.LINE_AA)
        tiles.append(cv)

    W = sum(t.shape[1] for t in tiles) + 12 * len(tiles) + 8
    H = max(t.shape[0] for t in tiles) + 76
    sheet = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(sheet, "4 pieces joined by edge geometry - ranked by exposed sliver area", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
    x = 8
    for t in tiles:
        sheet[64:64 + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 12
    cv2.imwrite(str(outdir / "41_union_rank.png"), sheet)
    print(f"写出 {outdir / '41_union_rank.png'} {sheet.shape[1]}x{sheet.shape[0]}")

    best = ranked[0][2]
    (outdir / "best_union.json").write_text(json.dumps(
        {"holes_pct": ranked[0][1]["holes_pct"], "overlap_pct": ranked[0][1]["overlap_pct"],
         "placed": best["placed"], "polys": best["polys"]}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
