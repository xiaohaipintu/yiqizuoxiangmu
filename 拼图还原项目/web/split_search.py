"""补充搜索：允许「短边顶在长边中间」（T 型接缝 / 边被两块共用）。

beam_search.py 做不到这种拼法，因为它有两条硬限制：
  1. 一条边只能用一次（`if ei in used[pid]: continue`）—— 长边被两块共用就无解；
  2. 只能端点到端点对齐（两种 anchor），没有「沿长边滑动」的自由度。

矩形的真实切法经常是「短边顶在长边中间」，所以候选里永远看不到 A2↔B4 这种相邻关系。
这里用同样的 Piece/world 原语，把两个限制都放开：

  * 短边（Lb）贴长边（La）时，允许沿长边滑动：位移取 {0, (La-Lb)/2, La-Lb} 三档
  * 同一条长边允许被多个碎片共用（只要不重叠）
  * 其它约束照旧：不镜像、不重叠（shapely 精确算）、公共长度 ≥ cov×短边

输出格式与 beam_solutions.json 完全一致，上层（重排/打分/渲染）不用改。

用法：
    python web/split_search.py --pieces pieces_scaled.json --out <任务目录> --time 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assemble import Piece, place_edge, world      # noqa: E402
import geometry as G                               # noqa: E402


def gap_pct(polys) -> float:
    pts = np.vstack(polys).astype(np.float32)
    hull = abs(float(cv2.contourArea(cv2.convexHull(pts))))
    s = sum(G.polygon_area(p) for p in polys)
    return max(0.0, hull - s) / s * 100.0


def placements(piece_q, fi, aw, bw, extra_shift: float):
    """把 q 的边 fi 贴到 a→b 上，额外沿边方向平移 extra_shift。"""
    ang, t = place_edge(piece_q, fi, aw, bw, True)
    if abs(extra_shift) > 1e-9:
        u = (bw - aw) / max(float(np.linalg.norm(bw - aw)), 1e-9)
        t = (float(t[0]) + u[0] * extra_shift, float(t[1]) + u[1] * extra_shift)
    return ang, t


def search(pieces: dict, deadline: float, cov: float = 0.30,
           overlap_tol: float = 0.01, max_states: int = 4000):
    """返回完整拼法列表 [(gap, placed, polys)]。"""
    ids = list(pieces)
    results = []

    def expand(placed, polys, used_edges):
        """placed: [(pid, ang, tx, ty)]，返回可扩展的候选。"""
        out = []
        done = {p[0] for p in placed}
        for qid in ids:
            if qid in done:
                continue
            q = pieces[qid]
            for k, (pid, _, _, _) in enumerate(placed):
                p = pieces[pid]
                pw = polys[k]
                for ei in range(p.n):
                    aw, bw = pw[ei], pw[(ei + 1) % p.n]
                    La = float(np.linalg.norm(bw - aw))
                    if La < 1e-6:
                        continue
                    u = (bw - aw) / La
                    for fi in range(q.n):
                        Lb = float(q.edges[fi]["length"])
                        if Lb < 1e-6:
                            continue
                        # 短边贴长边才有「顶在中间」的意义；两档滑动 + 两端
                        slides = {0.0}
                        if La > Lb + 1e-6:
                            slides |= {(La - Lb) / 2.0, La - Lb}
                        for s in slides:
                            ang, t = placements(q, fi, aw, bw, s)
                            wq = world(q, ang, t)
                            qa = (G.rotation(ang) @ np.asarray(q.edges[fi]["a"], float)) \
                                + np.asarray(t, float)
                            qb = (G.rotation(ang) @ np.asarray(q.edges[fi]["b"], float)) \
                                + np.asarray(t, float)
                            proj = np.clip([((qa - aw) @ u), ((qb - aw) @ u)], 0, La)
                            shared = float(max(0.0, proj.max() - proj.min()))
                            if shared < cov * min(La, Lb):
                                continue
                            # 必须贴在目标边的另一侧
                            ca, cq = p.poly.mean(axis=0), wq.mean(axis=0)
                            sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
                            sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                            if sa * sq >= 0:
                                continue
                            ov = sum(G.poly_overlap_area(wq, pp) for pp in polys)
                            if ov > overlap_tol * q.area:
                                continue
                            out.append((placed + [(qid, ang, float(t[0]), float(t[1]))],
                                        polys + [wq]))
        return out

    seed = max(pieces, key=lambda i: pieces[i].area)
    states = [([(seed, 0.0, 0.0, 0.0)], [world(pieces[seed], 0.0, (0.0, 0.0))])]
    for level in range(2, len(pieces) + 1):
        nxt = []
        for placed, polys in states:
            nxt.extend(expand(placed, polys, None))
            if time.time() > deadline:
                break
        if not nxt:
            break
        # 去重（同样的 placed 顺序组合只留一个）+ 按凸包空隙排序取前 max_states
        nxt.sort(key=lambda s: gap_pct(s[1]))
        states = nxt[:max_states]
        if time.time() > deadline:
            break

    for placed, polys in states:
        if len(placed) == len(pieces):
            results.append((gap_pct(polys), placed, polys))
    results.sort(key=lambda x: x[0])
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--cov", type=float, default=0.30)
    ap.add_argument("--save", type=int, default=300)
    args = ap.parse_args()

    data = json.loads(Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
    deadline = time.time() + args.time
    found = search(pieces, deadline, cov=args.cov)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "split_solutions.json").write_text(json.dumps({
        "n": len(found),
        "solutions": [{"gap_pct": g, "placed": pl,
                       "polys": [p.tolist() for p in ps]}
                      for g, pl, ps in found[:args.save]],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"T型接缝搜索：找到 {len(found)} 个完整拼法，最好 缝 {found[0][0]:.2f}%" if found
          else "T型接缝搜索：没有完整拼法")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
