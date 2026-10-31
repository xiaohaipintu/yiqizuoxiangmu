"""把「等长的边」贴在一起：成功几种就排几种，全部放在同一张大图上。

规则（只用边长，不镜像/不翻面）：
    * 两块碎片各取一条边，长度相等（容差内，默认 5%，吸收拍照透视误差）；
    * 两条边共线、方向相反、整条重叠（公共长度 ≈ 较短那条边）；
    * 两块分别在共线两侧（凸贴合，不重叠）。

用法:
    python src/equal_edges.py --pieces out2/pieces.json --out out2 --tol 0.05
输出:
    out2/30_equal_edges.png    所有成功配对的并排大图（含三块闭环那一格）
    out2/30_equal_edges.json   每格配对信息
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
from assemble import Piece, Enum, world, place_edge, draw, score  # noqa: E402
import geometry as G  # noqa: E402


def enumerate_joins(pieces, tol=0.05, overlap_tol=0.01):
    joins = []
    ids = list(pieces)
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            pa, pb = pieces[ids[a_i]], pieces[ids[b_i]]
            for ei in range(pa.n):
                ea = pa.edges[ei]
                La = float(ea["length"])
                aw, bw = np.array(ea["a"], float), np.array(ea["b"], float)
                u = (bw - aw) / (np.linalg.norm(bw - aw) + 1e-12)
                for fj in range(pb.n):
                    eb = pb.edges[fj]
                    Lb = float(eb["length"])
                    diff = abs(La - Lb) / max(La, Lb)
                    if diff > tol:
                        continue
                    ang, t = place_edge(pb, fj, aw, bw, True)  # a 端贴目标 b 端 → 反向整条覆盖
                    wq = world(pb, ang, t)
                    qaw = (G.rotation(ang) @ np.array([eb["a"]], float).T).T[0] + t
                    qbw = (G.rotation(ang) @ np.array([eb["b"]], float).T).T[0] + t
                    shared = overlap_len_seg((aw, bw), (qaw, qbw))
                    if shared < 0.9 * min(La, Lb):
                        continue
                    # 两块必须分居这条边的两侧
                    ca, cq = pa.centroid, wq.mean(axis=0)
                    sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
                    sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                    if sa * sq >= 0:
                        continue
                    if G.convex_overlap_area(wq, pa.poly) > overlap_tol * pb.area:
                        continue
                    joins.append({
                        "a": pa.id, "b": pb.id, "ea": ei, "eb": fj,
                        "La": La, "Lb": Lb, "diff": diff, "shared": shared,
                        "poly_a": pa.poly.tolist(), "poly_b": wq.tolist(),
                    })
    joins.sort(key=lambda j: (round(j["diff"], 3), j["a"], j["ea"]))
    return joins


def overlap_len_seg(s1, s2) -> float:
    a1, b1 = np.asarray(s1[0], float), np.asarray(s1[1], float)
    a2, b2 = np.asarray(s2[0], float), np.asarray(s2[1], float)
    u = (b1 - a1) / (np.linalg.norm(b1 - a1) + 1e-12)
    t2 = [(p - a1) @ u for p in (a2, b2)]
    L1 = float(np.linalg.norm(b1 - a1))
    return float(max(0.0, min(max(t2), L1) - max(min(t2), 0.0)))


def make_tile(polys, ids, title, sub, color, box=540, head=36, foot=46):
    im = draw(polys, ids, size=box - 30, margin=12)
    h, w = im.shape[:2]
    canvas = np.full((head + h + foot, max(w + 20, 460), 3), 255, np.uint8)
    x0 = (canvas.shape[1] - w) // 2
    canvas[head:head + h, x0:x0 + w] = im
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, sub, (10, head + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "no mirror  |  edge-length match only", (10, head + h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out2/pieces.json")
    ap.add_argument("--out", default="out2")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--no-closure", action="store_true")
    ap.add_argument("--overview", action="store_true",
                    help="图首加一格：全部碎片（按原始相对位置，带编号），便于逐块核对")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
    joins = enumerate_joins(pieces, tol=args.tol)
    print(f"等长边配对成功 {len(joins)} 种（容差 {args.tol*100:.0f}%，不镜像）")
    for j in joins:
        print(f"  #{j['a']}e{j['ea']}({j['La']:.0f}px)  x  #{j['b']}e{j['eb']}({j['Lb']:.0f}px)"
              f"   Δ{j['diff']*100:.1f}%")

    # 每块在结果里出现了几次：0 次说明它在这条容差下没有任何等长伙伴
    cover = {pid: 0 for pid in pieces}
    for j in joins:
        cover[j["a"]] += 1
        cover[j["b"]] += 1
    print("各碎片在配对结果中出现的次数: " +
          "  ".join(f"#{pid}:{n}" for pid, n in cover.items()))
    for pid, n in cover.items():
        if n == 0:
            print(f"  ! 注意 #{pid} 在 {args.tol*100:.0f}% 容差下没有等长边伙伴，因此不会出现在配对图里")

    tiles = []
    if args.overview:
        tiles.append(make_tile([pieces[pid].poly for pid in pieces],
                               [pid for pid in pieces],
                               f"ALL {len(pieces)} PIECES (as photographed)",
                               "ids + measured edges are listed in pieces.json",
                               (40, 40, 40)))
    for j in joins:
        d = j["diff"] * 100
        col = (40, 130, 40) if d < 2.0 else (30, 90, 170)
        tiles.append(make_tile([np.array(j["poly_a"]), np.array(j["poly_b"])], [j["a"], j["b"]],
                               f"{j['a']}e{j['ea']}  x  {j['b']}e{j['eb']}",
                               f"{j['La']:.0f} px  vs  {j['Lb']:.0f} px    diff {d:.1f}%", col))

    # 三块能不能只用等长边全部接上（闭环）
    closure = None
    if len(pieces) >= 3 and not args.no_closure:
        en = Enum(list(pieces.values()), min_cov=0.9, overlap_tol=0.004)
        results = en.run()
        print(f"三块（或全部）用整条等长边接上的组合：{len(results)} 种")
        if results:
            scored = sorted(((score(*r), r) for r in results), key=lambda x: x[0]["holes_pct"])
            s, r = scored[0]
            print(f"  最佳闭环：空穴率 {s['holes_pct']:.2f}%  外轮廓 {s['hull_perimeter']:.0f}px  "
                  f"{s['hull_vertices']} 顶点")
            closure = {"tile": make_tile(r[1], [p[0] for p in r[0]],
                                         "ALL 3 JOINED (closed loop)",
                                         f"holes {s['holes_pct']:.2f}%   outline {s['hull_perimeter']:.0f}px   "
                                         f"{s['hull_vertices']} vertices",
                                         (150, 60, 30)) if s["holes_pct"] < 3 else None,
                       "holes_pct": s["holes_pct"], "poly": [p.tolist() for p in r[1]],
                       "ids": [p[0] for p in r[0]]}
            if closure["tile"] is not None:
                tiles.append(closure["tile"])
                big = draw(r[1], [p[0] for p in r[0]], size=1100, margin=40)
                cv2.imwrite(str(outdir / "31_closed.png"), big)
                print(f"写出 {outdir / '31_closed.png'} {big.shape[1]}x{big.shape[0]}")

    cols = min(args.cols, max(1, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    tw = max(t.shape[1] for t in tiles) + 12
    th = max(t.shape[0] for t in tiles) + 12
    sheet = np.full((96 + rows * th, 24 + cols * tw, 3), 255, np.uint8)
    cv2.putText(sheet, f"equal-length edge joins - {len(joins)} successful pairs (no mirror, tol {args.tol*100:.0f}%)",
                (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(sheet, "green = diff < 2% (solid match)    blue = 2% ~ tol    brown = all three closed by the same rule",
                (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (95, 95, 95), 1, cv2.LINE_AA)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[86 + r * th:86 + r * th + im.shape[0], 12 + c * tw:12 + c * tw + im.shape[1]] = im
    cv2.imwrite(str(outdir / "30_equal_edges.png"), sheet)
    print(f"写出 {outdir / '30_equal_edges.png'} {sheet.shape[1]}x{sheet.shape[0]}")

    payload = {"tol": args.tol,
               "joins": [{k: v for k, v in j.items() if not k.startswith("poly")} for j in joins]}
    if closure and closure["tile"] is not None:
        payload["closure"] = {"holes_pct": closure["holes_pct"],
                              "ids": closure["ids"], "poly": closure["poly"]}
    (outdir / "30_equal_edges.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


