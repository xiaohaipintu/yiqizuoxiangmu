"""按「整体」拼接：#1+#3 作为整体 A 固定，#4+#2 作为整体 B 绕自身质心旋转 180° 后对接 A。

做法（只让 B 转、只让 B 平移，不加别的自由度）：
  1. 从已有解里取每块的世界位姿；
  2. 把 B 里每块绕 B 的联合质心转 180°（位姿更新：ang += 180，t' = 2c - t）；
  3. 枚举 A 的边 × B 的边，要求两段边方向近似反向，用中点对齐求出平移 d，
     逐块评估 A∪B 的「共线贴合总长」和「叠压面积」，取贴合最长、不叠压的 d；
  4. 把结果渲染成图（带 #N 编号）。

用法：
    python 纹理拼合/tools/join_groups.py --solution 纹理拼合/手撕拼图/solution_纯形状.json \
           --group-a 1,3 --group-b 4,2 --out 纹理拼合/手撕拼图
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
from handmade_puzzle import imread_u, imwrite_u, render, segment, draw_labels  # noqa: E402


# --------------------------------------------------------------------------
# 几何小工具
# --------------------------------------------------------------------------
def edges_of(poly: np.ndarray):
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


def seg_pair(a, b):
    """两条线段：返回到彼此直线的最大偏差、投影重叠长度。"""
    p, q = a
    r, s = b
    u = q - p
    L = float(np.linalg.norm(u))
    if L < 1e-9:
        return 1e9, 0.0
    u = u / L
    dev = max(abs(u[0] * (r[1] - p[1]) - u[1] * (r[0] - p[0])),
              abs(u[0] * (s[1] - p[1]) - u[1] * (s[0] - p[0])))
    t = ((r - p) @ u, (s - p) @ u)
    ov = max(0.0, min(max(t), L) - max(min(t), 0.0))
    return dev, ov


def shared_contact(polys_a, polys_b, tol: float = 0.6) -> float:
    """A、B 两组多边形之间的共线贴合总长（逐条边量，避开 GEOS 近共线漏判）。"""
    total = 0.0
    for pa in polys_a:
        for ea in edges_of(pa):
            for pb in polys_b:
                for eb in edges_of(pb):
                    dev, ov = seg_pair(ea, eb)
                    if dev < tol and ov > 5:
                        total += ov
    return total


def overlap_area(polys_a, polys_b) -> float:
    ua = unary_union([SPoly(p).buffer(0) for p in polys_a])
    ub = unary_union([SPoly(p).buffer(0) for p in polys_b])
    return float(ua.intersection(ub).area)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="整体拼接：A 固定，B 旋转 180° 后对接")
    ap.add_argument("--image", default="手撕拼图.jpg")
    ap.add_argument("--solution", default="纹理拼合/手撕拼图/solution_纯形状.json")
    ap.add_argument("--group-a", default="1,3")
    ap.add_argument("--group-b", default="4,2")
    ap.add_argument("--rotate", type=float, default=180.0)
    ap.add_argument("--refine", type=float, default=8.0,
                    help="在整体旋转基础上，允许贴边时的额外微调角度上限（度）")
    ap.add_argument("--out", default="纹理拼合/手撕拼图")
    ap.add_argument("--name", default="4_整体拼接效果图.jpg")
    ap.add_argument("--compare", action="store_true",
                    help="额外输出「原图 / 拼接前 / 拼接后」三图对比")
    args = ap.parse_args()

    group_a = [int(x) for x in args.group_a.split(",")]
    group_b = [int(x) for x in args.group_b.split(",")]

    src = imread_u(Path(args.image))
    cells, _ = segment(src, 100, 20000, 0.02)
    data = to_pieces_json(cells)
    pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}

    sol = json.loads(Path(args.solution).read_text(encoding="utf-8"))
    poses = {int(it[0]): [float(it[1]), float(it[2]), float(it[3])] for it in sol["placed"]}
    poses0 = {p: list(v) for p, v in poses.items()}      # 拼接前的原始位姿（用于对比图）
    print("输入解：", {k: ["%.1f°" % v[0]] for k, v in poses.items()})

    # 1) B 绕自身联合质心旋转
    wb = [world(pieces[p], poses[p][0], (poses[p][1], poses[p][2])) for p in group_b]
    cB = unary_union([SPoly(p).buffer(0) for p in wb]).centroid
    th = np.deg2rad(args.rotate)
    print("B 的联合质心 (%.1f, %.1f)，整体旋转 %.0f°" % (cB.x, cB.y, args.rotate))
    for p in group_b:
        poses[p][0] += args.rotate                   # ang += 180
        d = np.array([poses[p][1], poses[p][2]]) - np.array([cB.x, cB.y])
        nd = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ d
        poses[p][1], poses[p][2] = (nd + np.array([cB.x, cB.y])).tolist()

    def world_polys(group):
        return {p: world(pieces[p], poses[p][0], (poses[p][1], poses[p][2])) for p in group}

    def rotate_group(ps: dict, group, delta: float, center) -> None:
        """把 group 整体绕 center 旋转 delta 度（位姿更新）。"""
        th = np.deg2rad(delta)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        c = np.asarray(center, float)
        for p in group:
            ps[p][0] += delta
            d = np.array(ps[p][1:], float) - c
            nd = R @ d + c
            ps[p][1], ps[p][2] = float(nd[0]), float(nd[1])

    base = {p: list(v) for p, v in poses.items()}      # 旋转 180° 后的基准
    wa = world_polys(group_a)

    best = None
    for ea in [e for p in group_a for e in edges_of(wa[p])]:
        va = ea[1] - ea[0]
        La = float(np.linalg.norm(va))
        if La < 40:
            continue
        ua = va / La
        ma = (ea[0] + ea[1]) / 2
        for pid in group_b:
            for eb in edges_of(world_polys(group_b)[pid]):
                vb = eb[1] - eb[0]
                Lb = float(np.linalg.norm(vb))
                if Lb < 40 or min(La, Lb) / max(La, Lb) < 0.35:
                    continue
                ub = vb / Lb
                # 想要 R(δ)·ub = -ua（反向贴合）
                delta = (np.degrees(np.arctan2(-ua[1], -ua[0]) - np.arctan2(ub[1], ub[0])) + 180) % 360 - 180
                if abs(delta) > args.refine:
                    continue
                trial = {p: list(v) for p, v in base.items()}
                rotate_group(trial, group_b, delta, (cB.x, cB.y))
                wt = {p: world(pieces[p], trial[p][0], (trial[p][1], trial[p][2])) for p in group_b}
                # 平移：让这条边中点对上
                ebw = None
                for e2 in edges_of(wt[pid]):
                    e2m = (e2[0] + e2[1]) / 2
                    if np.linalg.norm(e2m - (eb[0] + eb[1]) / 2) < 1e-6:
                        ebw = e2
                        break
                if ebw is None:
                    ebw = min(edges_of(wt[pid]),
                              key=lambda e2: np.linalg.norm((e2[0] + e2[1]) / 2 - (eb[0] + eb[1]) / 2))
                dsh = ma - (ebw[0] + ebw[1]) / 2
                for p in group_b:
                    trial[p][1] += float(dsh[0])
                    trial[p][2] += float(dsh[1])
                wts = [world(pieces[p], trial[p][0], (trial[p][1], trial[p][2])) for p in group_b]
                sh = shared_contact(list(wa.values()), wts)
                ov = overlap_area(list(wa.values()), wts)
                score = sh - ov * 2.0
                if best is None or score > best[0]:
                    best = (score, trial, pid, sh, ov, delta)

    if best is None:
        print("没有找到可对接的边对（可放宽 --refine）")
        return 1
    _score, trial, via, sh, ov, delta = best
    print("最佳对接：B 的 #%d 附近边，额外微调 %.2f°，共线贴合 %.1f px，叠压 %.1f px²"
          % (via, delta, sh, ov))
    poses.update(trial)

    polys = [world(pieces[p], poses[p][0], (poses[p][1], poses[p][2])) for p in sorted(poses)]
    order = sorted(poses)
    new_sol = {"placed": [[p, poses[p][0], poses[p][1], poses[p][2]] for p in order],
               "polys": [q.tolist() for q in polys]}

    # 4) 验收指标
    wa = [world(pieces[p], poses[p][0], (poses[p][1], poses[p][2])) for p in group_a]
    wb = [world(pieces[p], poses[p][0], (poses[p][1], poses[p][2])) for p in group_b]
    cross = shared_contact(wa, wb)
    inner_a = shared_contact([wa[0]], [wa[1]]) if len(wa) > 1 else 0.0
    inner_b = shared_contact([wb[0]], [wb[1]]) if len(wb) > 1 else 0.0
    allp = [SPoly(q).buffer(0) for q in polys]
    u = unary_union(allp)
    total = sum(p.area for p in allp)
    rect = u.minimum_rotated_rectangle
    print()
    print("A 内部贴合 %.1f px    B 内部贴合 %.1f px    跨组贴合 %.1f px"
          % (inner_a, inner_b, cross))
    print("并集：重叠 %.2f%%  矩形贴合 %.1f%%  连通块 %s"
          % ((total - u.area) / total * 100, u.area / rect.area * 100, u.geom_type))

    # 5) 出图
    outdir = Path(args.out)
    effect = render(src, pieces, new_sol, labels=True)
    imwrite_u(outdir / args.name, effect)
    (outdir / ("solution_整体拼接.json")).write_text(
        json.dumps(new_sol, ensure_ascii=False, indent=2), encoding="utf-8")
    print("效果图 →", outdir / args.name)

    # 6) 可选：出「原图 / 拼接前 / 拼接后」三图对比
    if args.compare:
        old = render(src, pieces, {"placed": [[p, *poses0[p]] for p in sorted(poses0)],
                                   "polys": [world(pieces[p], poses0[p][0],
                                                   (poses0[p][1], poses0[p][2])).tolist()
                                             for p in sorted(poses0)]}, labels=True)
        labelled = draw_labels(src, cells)
        H = 600

        def fit(im, h=H):
            s = h / im.shape[0]
            return cv2.resize(im, (int(im.shape[1] * s), h), interpolation=cv2.INTER_AREA)
        panels = [("original", fit(labelled)), ("before", fit(old)), ("after (B +180°)", fit(effect))]
        gap = 14
        W = sum(p.shape[1] for _, p in panels) + gap * (len(panels) + 1)
        canvas = np.full((H + 54, W, 3), 16, np.uint8)
        x = gap
        for title, im in panels:
            canvas[46:46 + im.shape[0], x:x + im.shape[1]] = im
            cv2.putText(canvas, title, (x, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)
            x += im.shape[1] + gap
        imwrite_u(outdir / "5_对比_整体拼接.jpg", canvas)
        print("对比图 →", outdir / "5_对比_整体拼接.jpg")
    return 0

if __name__ == "__main__":
    sys.exit(main())
