#!/usr/bin/env python3
"""按「边长比例」把 3 个三角形一一匹配到凹形多边形的 3 个缺口。

用户需求（本模块的唯一目标）
----------------------------
1. 匹配依据是**比例**（尺度无关），不是绝对像素长度——三角形允许整体缩放后嵌进缺口；
2. **允许误差**，用 ``--tol-ratio`` / ``--tol-deg`` 控制，超差会被标记；
3. **每个三角形只能对应一个缺口**——三角形↔缺口构成双射（bijection），
   不允许两个三角形抢同一个缺口，也不允许一个缺口空着。

算法
----
* **缺口**：凹形多边形上的每个凹顶点（reflex）处，两条邻边 ``cur-prev`` 与
  ``next-cur`` 夹出的 V 形空腔。缺口特征 = 两边长度之比 + 张角。
* **三角形**：小碎片顶点常被圆角/毛边分裂成 4~5 个伪顶点（README 第 6 节），
  先取「最大面积内接三角形」还原真实 3 角。
* **代价**：``|log(缺口比 / 三角比)|`` 与 ``|Δ角|`` 分别按容差归一后加权。
* **双射求解**：3 缺口 × 3 三角形 × 3 角 × 2 边序，枚举 ``3! × 3^3 × 2`` 种组合，
  取总代价最小——因为穷举，结果是该模型下的**全局最优**，且天然满足
  「每三角形只对应一个缺口」。
* **校验**：把三角形按最小二乘相似变换放入（**不含镜像**，与项目 angle_match 一致），
  再用栅格采样统计重叠率（三角形压进凹形实体）与缺口率（凸包内没被补满的空位）。

用法::

    python src/notch_ratio_match.py --pieces out5/pieces_scaled.json --out out5
    python src/notch_ratio_match.py --pieces out4/pieces_scaled.json --out out4 --tol-ratio 0.2

产出：``<out>/notch_ratio_match.json`` 与 ``<out>/60_notch_ratio_match.svg``。

本模块刻意只用标准库（math/json/itertools），不依赖 numpy/cv2/shapely，
以便在没有装依赖的机器上也能直接跑通。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys

# 控制台可能不是 UTF-8（Windows 默认 cp936），避免打印中文时崩掉
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------- 几何原语
def cross(o, a, b):
    """(a-o) × (b-o) 的二维叉积。"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def signed_area(poly):
    """带符号面积：>0 逆时针，<0 顺时针。"""
    n = len(poly)
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
        for i in range(n)
    )


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def angle_deg(prev, cur, nxt):
    """顶点 cur 处、向量 (prev-cur) 与 (nxt-cur) 的夹角，0~180°。

    对凸顶点即内角；对凹（reflex）顶点即「缺口张角」（空腔那一侧的 V 角）。
    """
    ux, uy = prev[0] - cur[0], prev[1] - cur[1]
    vx, vy = nxt[0] - cur[0], nxt[1] - cur[1]
    nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(c))


def polygon_area(poly):
    return abs(signed_area(poly))


def reflex_indices(poly):
    """返回所有凹（reflex）顶点的下标。"""
    n = len(poly)
    s = 1.0 if signed_area(poly) > 0 else -1.0
    return [i for i in range(n)
            if cross(poly[(i - 1) % n], poly[i], poly[(i + 1) % n]) * s < 0]


def convex_hull(points):
    """Andrew monotone chain，返回逆时针凸包顶点。"""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(reversed(pts))
    return [list(p) for p in lower[:-1] + upper[:-1]]


def point_in_polygon(pt, poly, eps=0.0):
    """射线法；eps 用于把边界点算作内部（栅格采样时避免边界噪声）。"""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def similarity_fit(src, dst):
    """最小二乘相似变换 p' = w*p + t（复数域，天然不含镜像）。

    返回 (w, t)，其中 w = scale * e^{i·theta}；残差由调用方按需计算。
    """
    n = len(src)
    zs = [complex(p[0], p[1]) for p in src]
    zd = [complex(p[0], p[1]) for p in dst]
    mzs = sum(zs) / n
    mzd = sum(zd) / n
    num = sum((z - mzs) * (d - mzd).conjugate() for z, d in zip(zs, zd))
    den = sum(abs(z - mzs) ** 2 for z in zs)
    w = num / den if den > 1e-12 else complex(1, 0)
    t = mzd - w * mzs
    return w, t


def apply_similarity(poly, w, t):
    return [[(w * complex(p[0], p[1]) + t).real, (w * complex(p[0], p[1]) + t).imag]
            for p in poly]


def similarity_rms(src, dst, w, t):
    """相似变换后的均方根残差（像素）。"""
    tot = 0.0
    for p, q in zip(src, dst):
        zp = w * complex(p[0], p[1]) + t
        tot += abs(zp - complex(q[0], q[1])) ** 2
    return math.sqrt(tot / len(src))


# ------------------------------------------------------------- 特征提取
def extract_notches(poly):
    """凹形多边形的所有缺口（每个凹顶点一个）。"""
    notches = []
    n = len(poly)
    for i in reflex_indices(poly):
        prev = poly[(i - 1) % n]
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        la = dist(prev, cur)   # 流入边
        lb = dist(cur, nxt)    # 流出边
        if la < 1e-9 or lb < 1e-9:
            continue
        notches.append({
            "notch_index": len(notches),
            "vertex_index": i,
            "corner": cur,
            "prev": prev,
            "next": nxt,
            "side_in": la,
            "side_out": lb,
            "ratio": la / lb,
            "angle_deg": angle_deg(prev, cur, nxt),
            "mouth_chord": dist(prev, nxt),
        })
    return notches


def max_area_triangle(poly):
    """从多边形顶点里挑 3 个，取面积最大的组合（还原被伪顶点污染的真实三角形）。

    返回按原始下标升序排列的 3 个顶点，保证环序不被破坏。
    """
    if len(poly) == 3:
        return list(range(3)), [list(p) for p in poly]
    best_idx, best_area = None, -1.0
    for combo in itertools.combinations(range(len(poly)), 3):
        a, b, c = (poly[k] for k in combo)
        area = abs(cross(a, b, c)) * 0.5
        if area > best_area:
            best_area, best_idx = area, combo
    idx = list(best_idx)
    return idx, [[*poly[k]] for k in idx]


def triangle_corners(verts):
    """三角形的 3 个角：每角的两条邻边长度、比例与内角。"""
    out = []
    n = len(verts)
    for i in range(n):
        prev = verts[(i - 1) % n]
        cur = verts[i]
        nxt = verts[(i + 1) % n]
        ta = dist(prev, cur)
        tb = dist(cur, nxt)
        out.append({
            "corner": i,
            "vertex": cur,
            "prev": prev,
            "next": nxt,
            "side_in": ta,
            "side_out": tb,
            "ratio": ta / tb if tb > 1e-12 else float("inf"),
            "angle_deg": angle_deg(prev, cur, nxt),
            "exact_angle_deg": exact_triangle_angle(verts, i),
        })
    return out


def exact_triangle_angle(verts, i):
    """三角形真实内角（顶点序列已被还原成真三角形，直接用三点夹角）。"""
    n = len(verts)
    return angle_deg(verts[(i - 1) % n], verts[i], verts[(i + 1) % n])


# --------------------------------------------------------------- 配对代价
def pair_cost(notch, corner, tol_ratio, tol_deg):
    """单个 (缺口, 三角角) 的归一化代价。

    比例项用 |log| 使「两倍」与「一半」等量惩罚；角度项为绝对差。
    边序（哪条边对哪条边）取更优的一侧，属于旋转自由，不是镜像。
    """
    a_err = abs(corner["side_in"] / notch["side_in"] - 1.0)
    b_err = abs(corner["side_out"] / notch["side_out"] - 1.0)
    swapped_a = abs(corner["side_out"] / notch["side_in"] - 1.0)
    swapped_b = abs(corner["side_in"] / notch["side_out"] - 1.0)
    fwd = 0.5 * (a_err + b_err)
    swp = 0.5 * (swapped_a + swapped_b)
    shape_err = min(fwd, swp)

    # 比例误差必须跟着边序一起考虑倒置：corner.ratio = 邻边1/邻边2，
    # 若实际是反着贴（order == "swapped"），等价比例就是它的倒数。
    # 只比原值会把「2.296 对上 2.348」误判成「0.4355 对上 2.348」，
    # 误差从 0.02 虚高到 1.69，进而选出一个角度差 44° 的荒谬配对。
    n_ratio = max(notch["ratio"], 1e-12)
    c_ratio = max(corner["ratio"], 1e-12)
    log_fwd = abs(math.log(n_ratio / c_ratio))   # 正向：邻边比 = 缺口边比
    log_swp = abs(math.log(n_ratio * c_ratio))   # 反向：邻边比 = 缺口边比的倒数

    # order 必须与 ratio_err 用同一判据，否则「判代价时按倒置算、摆放时按正向转」，
    # 三角形会被转到错误方向，角点偏离缺口顶点上百像素。
    if log_fwd <= log_swp:
        order, ratio_err = "forward", log_fwd
    else:
        order, ratio_err = "swapped", log_swp

    d_ang = abs(notch["angle_deg"] - corner["angle_deg"])

    bad_ratio = ratio_err / tol_ratio
    bad_ang = d_ang / tol_deg
    # 两个准则分别归一后取「较差的那一个」作为主代价。
    # 若改成加权求和，比例项的量纲会压过角度项，出现「宁可角度差 44° 也要
    # 比例吻合 0.09」这种明显不合理的指派。
    cost = max(bad_ratio, bad_ang)
    return {
        "cost": cost,
        "bad_ratio": bad_ratio,
        "bad_angle": bad_ang,
        "order": order,
        "ratio_err": ratio_err,
        "angle_err_deg": d_ang,
        "length_shape_err": shape_err,
        "within_tol": ratio_err <= tol_ratio and d_ang <= tol_deg,
    }


def place_triangle(tri_verts, corner_idx, notch, order):
    """把三角形放进缺口，返回 (placed_poly, w, t, rms)。

    这里不用 3 点最小二乘，而是**把该角强制钉在缺口顶点上**：
    物理上三角形就是「角顶进缺口、两条边贴着缺口两条壁」，所以角点必须重合，
    误差只能体现在两条边上。做法是拿一条边定出完整的相似变换 w（旋转+缩放，
    复数除法天然不含镜像），另一条边的残差即 rms。两条边各试一次，取残差小的。
    """
    n = len(tri_verts)
    cur = complex(*tri_verts[corner_idx])
    prev = complex(*tri_verts[(corner_idx - 1) % n])
    nxt = complex(*tri_verts[(corner_idx + 1) % n])
    n_cur = complex(*notch["corner"])
    n_prev = complex(*notch["prev"])
    n_next = complex(*notch["next"])

    if order == "forward":
        options = [((prev, n_prev), (nxt, n_next)), ((prev, n_next), (nxt, n_prev))]
    else:
        options = [((prev, n_next), (nxt, n_prev)), ((prev, n_prev), (nxt, n_next))]

    best = None
    for (a, na), (b, nb) in options:
        if abs(a - cur) < 1e-9:
            continue
        w = (na - n_cur) / (a - cur)           # 角点精确重合 + 该边精确贴合
        t = n_cur - w * cur
        rms = abs((w * b + t) - nb)            # 另一条边落点误差
        if best is None or rms < best[0]:
            best = (rms, w, t)
    rms, w, t = best
    placed = apply_similarity([[p[0], p[1]] for p in tri_verts], w, t)
    return placed, w, t, rms


def solve_assignment(triangles, notches, tol_ratio, tol_deg, top_k=3):
    """穷举 三角形→缺口 的双射，取总代价最小（全局最优，天然一一对应）。

    triangles: [{"label":..., "verts":[...], "corners":[...]}, ...]
    notches:   extract_notches() 的输出
    返回 best: {"assign": {tri_idx: notch_idx}, "total": float, "detail":[...]}
    """
    n = len(triangles)
    if n != len(notches):
        raise SystemExit(
            f"数量不匹配：{n} 个三角形 vs {len(notches)} 个缺口——"
            "「每个三角形只对应一个缺口」要求两者数量相等。"
        )

    # 预计算每个 (三角形, 角) 对每个缺口的代价
    table = {}
    for ti, tri in enumerate(triangles):
        for ci, corner in enumerate(tri["corners"]):
            for ni, notch in enumerate(notches):
                table[(ti, ci, ni)] = pair_cost(notch, corner, tol_ratio, tol_deg)

    cands = []
    # 缺口的一个排列 = 双射；每个三角形再选一个角（3 种），穷举 3! × 3^3 × 2 边序
    for perm in itertools.permutations(range(len(notches)), n):
        for corners in itertools.product(range(3), repeat=n):
            total = 0.0
            worst = 0.0
            within = 0
            for ti in range(n):
                pc = table[(ti, corners[ti], perm[ti])]
                total += pc["cost"]
                worst = max(worst, pc["cost"])
                within += 1 if pc["within_tol"] else 0
            cands.append({
                "total": total,
                "worst": worst,
                "within": within,
                "assign": {ti: perm[ti] for ti in range(n)},
                "corner_of": {ti: corners[ti] for ti in range(n)},
            })
    # 排序目标严格对应需求：先让「落进容差的匹配数」最多（每个三角形都要有归宿），
    # 再让总误差最小，最后比最差单项。
    cands.sort(key=lambda c: (-c["within"], c["total"], c["worst"]))
    best = cands[0]
    best["table"] = table
    best["within"] = cands[0]["within"]
    best["worst"] = cands[0]["worst"]
    best["alternatives"] = cands[1:1 + top_k]
    return best


# ------------------------------------------------------------------ 校验
def validate_coverage(piece_poly, placed, step_target=320):
    """栅格采样：相对「凹形凸包」统计 重叠率 / 缺口率 / 外溢率。"""
    hull = convex_hull(piece_poly)
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    step = max(1.0, max(x1 - x0, y1 - y0) / step_target)

    n_hull = n_piece = n_tri = n_overlap = n_gap = n_outside = 0
    y = y0
    while y <= y1:
        x = x0
        while x <= x1:
            pt = (x, y)
            in_hull = point_in_polygon(pt, hull)
            if in_hull:
                n_hull += 1
                in_piece = point_in_polygon(pt, piece_poly)
                in_tri = any(point_in_polygon(pt, t) for t in placed)
                if in_piece:
                    n_piece += 1
                if in_tri:
                    n_tri += 1
                if in_piece and in_tri:
                    n_overlap += 1
                if not in_piece and not in_tri:
                    n_gap += 1
            elif any(point_in_polygon(pt, t) for t in placed):
                n_outside += 1
            x += step
        y += step

    pct = lambda k: (100.0 * k / n_hull) if n_hull else 0.0
    return {
        "sample_step_px": step,
        "hull_samples": n_hull,
        "hull_area_px2": polygon_area(hull),
        "overlap_pct": pct(n_overlap),
        "gap_pct": pct(n_gap),
        "outside_pct": pct(n_outside),
        "piece_pct": pct(n_piece),
        "triangles_pct": pct(n_tri),
    }


# ---------------------------------------------------------------- 可视化
def render_svg(path, piece_poly, triangles, matches, notches, hull, coverage, title):
    xs = [p[0] for p in piece_poly]
    ys = [p[1] for p in piece_poly]
    for t in triangles:
        for p in t["placed"]:
            xs.append(p[0])
            ys.append(p[1])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = max(x1 - x0, y1 - y0) * 0.06
    vb = f"{x0 - pad:.1f} {y0 - pad:.1f} {x1 - x0 + 2 * pad:.1f} {y1 - y0 + 2 * pad:.1f}"
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]

    def pts(poly):
        return " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in poly)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="1200" '
        f'font-family="system-ui,Microsoft YaHei,sans-serif">',
        '<rect x="-1e6" y="-1e6" width="2e6" height="2e6" fill="#0f172a"/>',
        f'<polygon points="{pts(hull)}" fill="none" stroke="#475569" '
        'stroke-width="3" stroke-dasharray="12 8"/>',
        f'<polygon points="{pts(piece_poly)}" fill="#334155" stroke="#94a3b8" stroke-width="2"/>',
    ]

    for k, t in enumerate(triangles):
        c = colors[k % len(colors)]
        lines.append(
            f'<polygon points="{pts(t["placed"])}" fill="{c}" fill-opacity="0.45" '
            f'stroke="{c}" stroke-width="3"/>'
        )
        cx = sum(p[0] for p in t["placed"]) / len(t["placed"])
        cy = sum(p[1] for p in t["placed"]) / len(t["placed"])
        m = next(mm for mm in matches if mm["triangle_label"] == t["label"])
        lines.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" fill="#f8fafc" font-size="34" '
            f'text-anchor="middle">T{t["label"]}→缺口{m["notch_index"]}</text>'
        )

    for nch in notches:
        c = nch["corner"]
        lines.append(
            f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="10" fill="#facc15" '
            'stroke="#0f172a" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{c[0] + 18:.1f}" y="{c[1] - 14:.1f}" fill="#facc15" '
            f'font-size="26">缺口{nch["notch_index"]}</text>'
        )

    lines.append(f'<text x="{x0 - pad:.1f}" y="{y0 - pad + 42:.1f}" fill="#e2e8f0" '
                 f'font-size="40">{title}</text>')
    lines.append(
        f'<text x="{x0 - pad:.1f}" y="{y1 + pad - 20:.1f}" fill="#94a3b8" font-size="26">'
        f'重叠 {coverage["overlap_pct"]:.2f}% · 缺口 {coverage["gap_pct"]:.2f}% · '
        f'外溢 {coverage["outside_pct"]:.2f}%</text>'
    )
    lines.append("</svg>")
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------- 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="按边长比例把 3 个三角形一一匹配到凹形的 3 个缺口")
    ap.add_argument("--pieces", required=True, help="pieces(_scaled).json 路径")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--tol-ratio", type=float, default=0.15, help="比例相对容差，默认 0.15")
    ap.add_argument("--tol-deg", type=float, default=10.0, help="角度容差（度），默认 10")
    args = ap.parse_args(argv)

    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = data["pieces"]

    # 1) 找凹形：凹顶点最多的那块，其次面积最大
    scored = [(len(reflex_indices(p["polygon"])), polygon_area(p["polygon"]), p) for p in pieces]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    n_reflex, _, concave = scored[0]
    if n_reflex == 0:
        raise SystemExit("没有找到凹形碎片（无凹顶点）。")
    others = [p for p in pieces if p is not concave]

    piece_poly = [[float(x), float(y)] for x, y in concave["polygon"]]
    notches = extract_notches(piece_poly)

    # 2) 其余碎片各自还原为真三角形
    triangles = []
    for p in others:
        idx, verts = max_area_triangle([[float(x), float(y)] for x, y in p["polygon"]])
        triangles.append({
            "label": p["label"],
            "source_vertices": len(p["polygon"]),
            "kept_vertex_indices": idx,
            "verts": verts,
            "corners": triangle_corners(verts),
        })

    print(f"[输入] {args.pieces}")
    print(f"[凹形] label={concave['label']}  顶点 {len(piece_poly)}  缺口 {len(notches)} 个")
    for nch in notches:
        print(f"       缺口{nch['notch_index']}: 顶点#{nch['vertex_index']} "
              f"边 {nch['side_in']:.1f} / {nch['side_out']:.1f}  "
              f"比例 {nch['ratio']:.4f}  张角 {nch['angle_deg']:.2f}°")
    print(f"[三角] {len(triangles)} 块")
    for t in triangles:
        cs = "  ".join(
            f"角{c['corner']}:{c['angle_deg']:.1f}°(比{c['ratio']:.4f})" for c in t["corners"]
        )
        print(f"       label={t['label']} 原{ t['source_vertices']}顶点→真三角  {cs}")

    # 3) 双射求解
    best = solve_assignment(triangles, notches, args.tol_ratio, args.tol_deg)

    # 4) 放入并校验
    matches, placed_all = [], []
    for ti, tri in enumerate(triangles):
        ni = best["assign"][ti]
        ci = best["corner_of"][ti]
        notch = notches[ni]
        corner = tri["corners"][ci]
        pc = pair_cost(notch, corner, args.tol_ratio, args.tol_deg)
        placed, w, t_, rms = place_triangle(tri["verts"], ci, notch, pc["order"])
        tri["placed"] = placed
        placed_all.append(placed)
        matches.append({
            "triangle_label": tri["label"],
            "notch_index": ni,
            "notch_vertex_index": notch["vertex_index"],
            "triangle_corner": ci,
            "edge_order": pc["order"],
            "scale": abs(w),
            "rotation_deg": math.degrees(math.atan2(w.imag, w.real)),
            "ratio_err": pc["ratio_err"],
            "angle_err_deg": pc["angle_err_deg"],
            "length_shape_err": pc["length_shape_err"],
            "rms_px": rms,
            "cost": pc["cost"],
            "bad_ratio": pc["bad_ratio"],
            "bad_angle": pc["bad_angle"],
            "within_tol": pc["within_tol"],
        })

    hull = convex_hull(piece_poly)
    coverage = validate_coverage(piece_poly, placed_all)

    # 5) 双射自检
    used_notches = [m["notch_index"] for m in matches]
    used_tris = [m["triangle_label"] for m in matches]
    bijection_ok = (
        len(set(used_notches)) == len(used_notches)
        and len(set(used_tris)) == len(used_tris)
        and sorted(set(used_notches)) == sorted(n["notch_index"] for n in notches)
    )

    print("\n[匹配结果] 三角形 → 缺口（双射）")
    print(f"  {'三角':<8}{'缺口':>4}  {'缩放':>7}  {'旋转':>8}  {'比例误差':>9}  "
          f"{'角度误差':>8}  {'RMS(px)':>8}  达标")
    for m in matches:
        print(f"  label{m['triangle_label']:<3}{m['notch_index']:>6}  {m['scale']:>7.3f}  "
              f"{m['rotation_deg']:>7.1f}°  {m['ratio_err']:>9.4f}  "
              f"{m['angle_err_deg']:>7.2f}°  {m['rms_px']:>8.2f}  "
              f"{'是' if m['within_tol'] else '否'}")
    print(f"\n[双射校验] 每三角形↔唯一缺口: {'通过' if bijection_ok else '失败'}"
          f"  未匹配缺口 {sorted(set(n['notch_index'] for n in notches) - set(used_notches)) or '无'}")
    print(f"[覆盖校验] 重叠 {coverage['overlap_pct']:.2f}%  "
          f"缺口 {coverage['gap_pct']:.2f}%  外溢 {coverage['outside_pct']:.2f}%  "
          f"(栅格步长 {coverage['sample_step_px']:.2f}px)")

    # 6) 输出
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "source": args.pieces,
        "image": data.get("image"),
        "tolerance": {"ratio": args.tol_ratio, "angle_deg": args.tol_deg},
        "concave_piece": {"label": concave["label"], "vertices": len(piece_poly)},
        "notches": [{k: v for k, v in n.items() if k not in ("prev", "next", "corner")}
                    | {"corner": n["corner"]} for n in notches],
        "triangles": [{
            "label": t["label"],
            "source_vertices": t["source_vertices"],
            "kept_vertex_indices": t["kept_vertex_indices"],
            "vertices": t["verts"],
            "corners": [{k: c[k] for k in ("corner", "side_in", "side_out", "ratio", "angle_deg")}
                        for c in t["corners"]],
        } for t in triangles],
        "matches": matches,
        "bijection_ok": bijection_ok,
        "unmatched_notches": sorted(set(n["notch_index"] for n in notches) - set(used_notches)),
        "unmatched_triangles": sorted(set(t["label"] for t in triangles) - set(used_tris)),
        "total_cost": best["total"],
        "coverage": coverage,
    }
    json_path = out_dir / "notch_ratio_match.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    svg_path = out_dir / "60_notch_ratio_match.svg"
    render_svg(svg_path, piece_poly, triangles, matches, notches, hull, coverage,
               f"比例匹配：{len(triangles)} 三角形 ↔ {len(notches)} 缺口")

    print(f"\n[输出] {json_path}")
    print(f"[输出] {svg_path}")
    return 0 if bijection_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
