"""按"指定边配对"精确拼接：把一组碎片当作刚体，旋转+平移到让两条指定边贴死。

适合"算法挑错了、但人工能指出正确边对"的情况 —— 不再让代价函数参与判断。

用法（把 #4、#2 这一组整体转动，让 #4 的 D 边贴到 #3 的 A 边）：
    python 纹理拼合/tools/fit_edges.py --solution 纹理拼合/手撕拼图/solution_形状加纹理.json \
           --pair 3A-4D --move 4,2 --out 纹理拼合/手撕拼图 --name 9_修正版_3A接4D.jpg

    --pair   目标边-移动边（顺序无所谓，脚本会自动判断哪边属于移动组）
    --move   允许移动/旋转的块号（其余保持不动）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon as SPoly
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))

from assemble import Piece, world  # noqa: E402
from irregular_cut import to_pieces_json  # noqa: E402
from handmade_puzzle import imread_u, imwrite_u, render, segment  # noqa: E402

LE = "ABCDEFGHIJ"


def parse_edge(name: str):
    digits = "".join(ch for ch in name if ch.isdigit())
    letters = "".join(ch for ch in name if ch.isalpha())
    if not digits or not letters:
        raise ValueError("边名要像 3A：%r" % name)
    return int(digits), LE.index(letters.upper())


def edge_seg(poly: np.ndarray, idx: int):
    n = len(poly)
    return poly[idx % n], poly[(idx + 1) % n]


def seg_collinear(a, b, tol: float = 0.6):
    u = a[1] - a[0]
    L = float(np.linalg.norm(u))
    if L < 1e-9:
        return 1e9, 0.0
    u = u / L
    dev = max(abs(u[0] * (b[0][1] - a[0][1]) - u[1] * (b[0][0] - a[0][0])),
              abs(u[0] * (b[1][1] - a[0][1]) - u[1] * (b[1][0] - a[0][0])))
    t = ((b[0] - a[0]) @ u, (b[1] - a[0]) @ u)
    ov = max(0.0, min(max(t), L) - max(min(t), 0.0))
    return dev, ov


def main() -> int:
    ap = argparse.ArgumentParser(description="按指定边对精确拼接")
    ap.add_argument("--image", default="手撕拼图.jpg")
    ap.add_argument("--solution", default="纹理拼合/手撕拼图/solution_形状加纹理.json")
    ap.add_argument("--pair", required=True, help="两条边，例如 3A-4D")
    ap.add_argument("--move", default="", help="允许移动的块号，逗号分隔；空=只动 --pair 里属于非目标边的块")
    ap.add_argument("--out", default="纹理拼合/手撕拼图")
    ap.add_argument("--name", default="9_修正版.jpg")
    ap.add_argument("--labels", action="store_true", help="同时在图上标边名")
    ap.add_argument("--zoom", type=float, default=0.0)
    args = ap.parse_args()

    src = imread_u(Path(args.image))
    cells, _ = segment(src, 100, 20000, 0.02)
    pieces = {int(d["id"]): Piece(d) for d in to_pieces_json(cells)["pieces"]}

    sol = json.loads(Path(args.solution).read_text(encoding="utf-8"))
    poses = {int(it[0]): [float(it[1]), float(it[2]), float(it[3])] for it in sol["placed"]}

    def wpolys():
        return {p: world(pieces[p], poses[p][0], (poses[p][1], poses[p][2])) for p in poses}

    e1, e2 = [s.strip() for s in args.pair.replace("=", "-").split("-") if s.strip()]
    p1, i1 = parse_edge(e1)
    p2, i2 = parse_edge(e2)
    print("配对：%s（块%d 的 %s 边）  ↔  %s（块%d 的 %s 边）"
          % (e1, p1, LE[i1], e2, p2, LE[i2]))

    if args.move:
        group = [int(x) for x in args.move.split(",")]
    else:
        group = [p2]
    print("允许移动的块：", group, " 其余保持不动")

    W = wpolys()
    sa = edge_seg(W[p1], i1)
    sb = edge_seg(W[p2], i2)

    # 让 sb 反向贴合 sa
    ua = (sa[1] - sa[0]) / (np.linalg.norm(sa[1] - sa[0]) + 1e-12)
    ub = (sb[1] - sb[0]) / (np.linalg.norm(sb[1] - sb[0]) + 1e-12)
    delta = float(np.degrees(np.arctan2(-ua[1], -ua[0]) - np.arctan2(ub[1], ub[0])))
    print("需要把移动组旋转 %.2f°" % delta)

    # 旋转中心：移动组的联合质心
    c = np.array(unary_union([SPoly(W[g]).buffer(0) for g in group]).centroid.coords[0])
    th = np.deg2rad(delta)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for g in group:
        poses[g][0] += delta
        d = np.array(poses[g][1:], float) - c
        nd = R @ d + c
        poses[g][1], poses[g][2] = float(nd[0]), float(nd[1])

    # 旋转后再算一次，平移让中点对齐
    W = wpolys()
    sb2 = edge_seg(W[p2], i2)
    dsh = (sa[0] + sa[1]) / 2 - (sb2[0] + sb2[1]) / 2
    for g in group:
        poses[g][1] += float(dsh[0])
        poses[g][2] += float(dsh[1])
    print("平移 (%.1f, %.1f) 使两条边中点重合" % (dsh[0], dsh[1]))

    # 结果核算
    W = wpolys()
    print()
    print("贴合关系（共线偏差<0.6px 且重叠>40px）：")
    ids = sorted(W)
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            for ei in range(len(W[ids[x]])):
                for ej in range(len(W[ids[y]])):
                    if np.linalg.norm(np.diff(np.vstack(edge_seg(W[ids[x]], ei)), axis=0)) < 20:
                        continue
                    dev, ov = seg_collinear(edge_seg(W[ids[x]], ei), edge_seg(W[ids[y]], ej))
                    if dev < 0.6 and ov > 40:
                        print("   %d%s  ↔  %d%s    共线贴合 %5.0f px  (偏差 %.3f)"
                              % (ids[x], LE[ei], ids[y], LE[ej], ov, dev))

    ps = [SPoly(W[i].__array__()).buffer(0) if False else SPoly(W[i]).buffer(0) for i in ids]
    u = unary_union(ps)
    total = sum(q.area for q in ps)
    rect = u.minimum_rotated_rectangle
    print()
    print("并集：重叠 %.2f%%   连通块 %s   最小外接矩形贴合 %.1f%%"
          % ((total - u.area) / total * 100, u.geom_type, u.area / rect.area * 100))

    new_sol = {"placed": [[p, poses[p][0], poses[p][1], poses[p][2]] for p in ids],
               "polys": [W[p].tolist() for p in ids]}
    outdir = Path(args.out)
    effect = render(src, {}, new_sol, labels=False)
    imwrite_u(outdir / args.name, effect)
    print("效果图 →", outdir / args.name)
    (outdir / "solution_修正版.json").write_text(
        json.dumps(new_sol, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.labels:
        S = args.zoom if args.zoom > 0 else max(1.0, 1400.0 / max(1, effect.shape[1]))
        out = cv2.resize(effect, (int(effect.shape[1] * S), int(effect.shape[0] * S)),
                         interpolation=cv2.INTER_CUBIC) if S > 1 else effect.copy()
        polys = [np.asarray(q, float) for q in new_sol["polys"]]
        off = np.vstack(polys).min(axis=0) - 24
        from handmade_puzzle import _put_label
        sc = max(0.8, min(1.6, out.shape[1] / 900.0))
        cols = [(60, 90, 240), (60, 200, 100), (240, 140, 60), (220, 80, 220)]
        for it, q in zip(new_sol["placed"], polys):
            pid = int(it[0])
            cen = q.mean(axis=0)
            for i in range(len(q)):
                a, b = q[i], q[(i + 1) % len(q)]
                if np.linalg.norm(b - a) < 20:
                    continue
                col = cols[i % len(cols)]
                cv2.line(out, tuple(np.round((a - off) * S).astype(int)),
                         tuple(np.round((b - off) * S).astype(int)), col,
                         max(2, int(round(3 * S))), cv2.LINE_AA)
                mid = (a + b) / 2
                u2 = cen - mid
                u2 = u2 / (np.linalg.norm(u2) + 1e-9)
                _put_label(out, "%d%s" % (pid, LE[i]),
                           int((mid + u2 * 26 - off)[0] * S),
                           int((mid + u2 * 26 - off)[1] * S), sc, col)
        nm = args.name.rsplit(".", 1)[0] + "_边编号.jpg"
        imwrite_u(outdir / nm, out)
        print("带边名 →", outdir / nm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
