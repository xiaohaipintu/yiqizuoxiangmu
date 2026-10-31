"""拼接搜索（不镜像）：允许部分贴合（一块的边可以只贴住另一块边的一段，即 T 型接缝）。

规则:
    * 只用边的几何：新块的一条边贴到已放块某条边的共线位置（反向、贴另一侧）；
      端点锚定有两种：新块 a 端贴目标 b 端 / 新块 b 端贴目标 a 端（后者对应"只贴一段"）；
    * 公共长度 ≥ cov × 较短边（默认 0.35）；
    * 不镜像/不翻面；不重叠（用 shapely 精确算，支持凹多边形）；
    * 逐层扩展 + beam 保留缝最小的若干状态。

用法:
    python src/beam_search.py --pieces out4/pieces.json --out out4 --beam 300 --cov 0.35
输出:
    out4/beam_solutions.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assemble import Piece, world, place_edge  # noqa: E402
import geometry as G  # noqa: E402


def gap_pct(polys):
    pts = np.vstack(polys).astype(np.float32)
    hull = abs(float(cv2.contourArea(cv2.convexHull(pts))))
    s = sum(G.polygon_area(p) for p in polys)
    return max(0.0, hull - s) / s * 100.0


def expand(pieces, placed, polys, used, qid, cov, overlap_tol):
    q = pieces[qid]
    out = []
    for k, (pid, _, _, _) in enumerate(placed):
        pw = polys[k]
        p = pieces[pid]
        for ei in range(p.n):
            if ei in used[pid]:
                continue
            aw, bw = pw[ei], pw[(ei + 1) % p.n]
            La = float(np.linalg.norm(bw - aw))
            if La < 1e-6:
                continue
            u = (bw - aw) / La
            for fi in range(q.n):
                eb = q.edges[fi]
                Lb = float(eb["length"])
                for anchor in (True, False):
                    ang, t = place_edge(q, fi, aw, bw, anchor)
                    wq = world(q, ang, t)
                    qaw = (G.rotation(ang) @ np.asarray(eb["a"], float)) + np.asarray(t, float)
                    qbw = (G.rotation(ang) @ np.asarray(eb["b"], float)) + np.asarray(t, float)
                    proj = np.clip([((qaw - aw) @ u), ((qbw - aw) @ u)], 0, La)
                    shared = float(max(0.0, proj.max() - proj.min()))
                    if shared < cov * min(La, Lb):
                        continue
                    ca, cq = p.poly.mean(axis=0), wq.mean(axis=0)
                    sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
                    sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                    if sa * sq >= 0:
                        continue
                    ov = sum(G.poly_overlap_area(wq, pp) for pp in polys)
                    if ov > overlap_tol * q.area:
                        continue
                    out.append((pid, ei, fi, ang, float(t[0]), float(t[1]), wq))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out4/pieces.json")
    ap.add_argument("--out", default="out4")
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--cov", type=float, default=0.35)
    ap.add_argument("--overlap", type=float, default=0.004)
    ap.add_argument("--time", type=float, default=180.0)
    ap.add_argument("--save", type=int, default=30,
                    help="最多保存多少个完整解到 beam_solutions.json（上层重排要看更多候选）")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
    deadline = time.time() + args.time

    seed = max(pieces, key=lambda i: pieces[i].area)
    states = [([(seed, 0.0, 0.0, 0.0)], [world(pieces[seed], 0.0, (0.0, 0.0))],
               {pid: set() for pid in pieces})]
    print(f"碎片 {len(pieces)}，起始块 #{seed}")
    for level in range(2, len(pieces) + 1):
        cand = []
        for placed, polys, used in states:
            done = {r[0] for r in placed}
            for qid in pieces:
                if qid in done:
                    continue
                for pid, ei, fi, ang, tx, ty, wq in expand(pieces, placed, polys, used,
                                                          qid, args.cov, args.overlap):
                    nu = {kk: set(v) for kk, v in used.items()}
                    nu[pid].add(ei)
                    nu[qid] = {fi}
                    cand.append((placed + [(qid, ang, tx, ty)], polys + [wq], nu,
                                 gap_pct(polys + [wq])))
            if time.time() > deadline:
                break
        if not cand:
            print(f"第 {level} 层：无可行扩展")
            break
        cand.sort(key=lambda s: s[3])
        states = [(a, b, c) for a, b, c, _ in cand[: args.beam]]
        print(f"第 {level} 层：候选 {len(cand)}，缝最大 {cand[0][3]:.2f}% / "
              f"保留 {len(states)}（最差 {cand[min(len(cand) - 1, args.beam - 1)][3]:.2f}%）")
        if time.time() > deadline:
            print("超时，停在当前层")
            break

    full = sorted([(gap_pct(polys), placed, polys) for placed, polys, used in states
                   if len(placed) == len(pieces)], key=lambda x: x[0])
    if full:
        print(f"完整拼法 {len(full)} 个，最好 缝 {full[0][0]:.2f}%")
    else:
        print("没有完整拼法")
    (outdir / "beam_solutions.json").write_text(json.dumps(
        {"n": len(full), "solutions": [{"gap_pct": g, "placed": pl,
                                        "polys": [p.tolist() for p in ps]}
                                       for g, pl, ps in full[:args.save]]}, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
