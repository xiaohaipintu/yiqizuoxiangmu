"""两两配对：只要一对边能接上就算成立，不要求所有碎片连成一个整体。

对每个碎片对，枚举「两块各取一条边 + 端点锚定」的所有贴合方式，
按「两条边长度差」优先、贴合长度其次排序，取每对的最佳结果截图。

用法:
    python src/pairwise.py --pieces out2/pieces.json --out out2
输出:
    out2/20_pairs.png       三对（1+2 / 1+3 / 2+3）的最佳贴合并排
    out2/21_best_pair.png   其中最精确的一对，大图
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assemble import Piece, world, place_edge, overlap_len, draw, PALETTE  # noqa: E402
import geometry as G  # noqa: E402


def best_join(a: Piece, b: Piece, overlap_tol=0.02):
    """枚举 a、b 的边对，返回最佳贴合（长度差最小优先）。"""
    best = None
    for ei in range(a.n):
        ea = a.edges[ei]
        La = float(ea["length"])
        A = np.array(a.poly, float)
        for fj in range(b.n):
            eb = b.edges[fj]
            Lb = float(eb["length"])
            for anchor in (True, False):
                ang, t = place_edge(b, fj, np.array(ea["a"], float), np.array(ea["b"], float), anchor)
                wb = world(b, ang, t)
                bqa = (G.rotation(ang) @ np.array([eb["a"]], float).T).T[0] + t
                bqb = (G.rotation(ang) @ np.array([eb["b"]], float).T).T[0] + t
                shared = overlap_len((ea["a"], ea["b"]), (bqa, bqb))
                if shared < 0.15 * min(La, Lb):
                    continue
                if G.convex_overlap_area(wb, A) > overlap_tol * b.area:
                    continue
                delta = abs(La - Lb) / La
                key = (round(delta, 4), -shared)
                if best is None or key < best[0]:
                    best = (key, ei, fj, ang, t, wb, La, Lb, shared)
    return best


def tile(pieces_xy, ids, size, title, subtitle):
    polys = [p for p in pieces_xy]
    im = draw(polys, ids, size=size, margin=16)
    h, w = im.shape[:2]
    canvas = np.full((h + 52, max(w, 360), 3), 255, np.uint8)
    x0 = (canvas.shape[1] - w) // 2
    canvas[46:46 + h, x0:x0 + w] = im
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (10, h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out2/pieces.json")
    ap.add_argument("--out", default="out2")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}

    rows = []
    best_overall = None
    best_key = None
    for i in range(1, len(pieces) + 1):
        for j in range(i + 1, len(pieces) + 1):
            a, b = pieces[i], pieces[j]
            r = best_join(a, b)
            if r is None:
                print(f"#{i}+#{j}: 没有可行的边贴合")
                continue
            key, ei, fj, ang, t, wb, La, Lb, shared = r
            rows.append((i, j, ei, fj, ang, t, wb, La, Lb, shared))
            print(f"#{i}+#{j}: 边 {i}e{ei}({La:.0f}px) ↔ {j}e{fj}({Lb:.0f}px)  "
                  f"长度差 {abs(La-Lb)/La*100:.1f}%  贴合段 {shared:.0f}px")
            if best_key is None or key < best_key:
                best_key = key
                best_overall = (r, i, j)

    # 并排图
    tiles = []
    for (i, j, ei, fj, ang, t, wb, La, Lb, shared) in rows:
        pa = pieces[i].poly
        tiles.append(tile([pa, wb], [i, j], 380,
                          f"#{i} + #{j}",
                          f"edge {i}e{ei} ({La:.0f}px)  x  {j}e{fj} ({Lb:.0f}px)   diff {abs(La-Lb)/La*100:.1f}%"))
    sheet = np.full((max(t.shape[0] for t in tiles), sum(t.shape[1] for t in tiles) + 10 * len(tiles), 3),
                    255, np.uint8)
    x = 0
    for t in tiles:
        sheet[: t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 10
    cv2.imwrite(str(outdir / "20_pairs.png"), sheet)
    print(f"写出 {outdir / '20_pairs.png'} {sheet.shape[1]}x{sheet.shape[0]}")

    if best_overall:
        r, i, j = best_overall
        key, ei, fj, ang, t, wb, La, Lb, shared = r
        big = draw([pieces[i].poly, wb], [i, j], size=1000, margin=30)
        cv2.imwrite(str(outdir / "21_best_pair.png"), big)
        print(f"写出 {outdir / '21_best_pair.png'} {big.shape[1]}x{big.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
