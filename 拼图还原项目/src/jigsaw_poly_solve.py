"""多片随机拼图求解（换算法版）——针对「几十片 + 任意旋转」的散落碎片。

旧流水线（beam_search/缺口双射）是为 2~7 片调的，几十片会组合爆炸、顶到 150s 上限还拼不出。
这里换一套思路，利用「拼图碎片是凸多边形、相邻两片共享同一条边（长度精确相等）」这个结构：

  1) Voronoi 把信纸切成 N 片凸多边形（测试用；真实场景这一步换成分割出的多边形）
  2) 每片随机旋转 + 随机平移散开（不重叠）——模拟拍照后碎片乱摆
  3) 求解：对每条边按「长度相等」找唯一对边（相邻两片共享边 → 长度零误差），
     把其中一片按刚体变换（纯旋转+平移，不镜像）贴到另一片边上，BFS 逐片长出来
  4) 渲染复原图；用 shapely 校验重叠/空洞

用法:
  python src/jigsaw_poly_solve.py --n 60 --seed 7 \
      --image 拼图还原项目/svg_tool/test_out/letter_h5.png --out 拼图还原项目/svg_tool/test_out/poly_solve
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from scipy.spatial import Voronoi

try:
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


# ---------------------------------------------------------------- 基础几何
def poly_area(p):
    p = np.asarray(p, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _signed_area(p):
    p = np.asarray(p, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def poly_centroid(p):
    p = np.asarray(p, float)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = cr.sum()
    if abs(a) < 1e-9:
        return p.mean(0)
    cx = ((x + xn) * cr).sum() / (3 * a)
    cy = ((y + yn) * cr).sum() / (3 * a)
    return np.array([cx, cy])


def _clip_half(poly, a, b, c):
    """保留 a*x+b*y+c<=0 的部分（Sutherland-Hodgman）。"""
    out = []
    n = len(poly)
    for i in range(n):
        p = poly[i]; q = poly[(i + 1) % n]
        dp = a * p[0] + b * p[1] + c
        dq = a * q[0] + b * q[1] + c
        if dp <= 0:
            out.append([p[0], p[1]])
        if (dp < 0 < dq) or (dq < 0 < dp):
            t = dp / (dp - dq)
            out.append([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
    return out


def clip_rect(poly, S):
    poly = _clip_half(poly, -1, 0, 0)       # x>=0
    poly = _clip_half(poly, 1, 0, -S)       # x<=S
    poly = _clip_half(poly, 0, -1, 0)       # y>=0
    poly = _clip_half(poly, 0, 1, -S)       # y<=S
    return poly


def convex_overlap(A, B, gap=0.0):
    """凸多边形 SAT：返回 True 表示相交（gap>0 时要求留间隙）。"""
    for P in (A, B):
        n = len(P)
        for i in range(n):
            p = P[i]; q = P[(i + 1) % n]
            nx, ny = -(q[1] - p[1]), (q[0] - p[0])
            da = A[:, 0] * nx + A[:, 1] * ny
            db = B[:, 0] * nx + B[:, 1] * ny
            if da.max() + gap <= db.min() or db.max() + gap <= da.min():
                return False
    return True


def transform(poly, R, t):
    return np.asarray(poly, float) @ R.T + t


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------- 切分
def voronoi_cells(seeds, S):
    far = np.array([[-1, -1], [2, -1], [2, 2], [-1, 2],
                    [-1, 0.5], [2, 0.5], [0.5, -1], [0.5, 2]], float) * S
    vor = Voronoi(np.vstack([seeds, far]))
    cells = []
    for i in range(len(seeds)):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            poly = [[0, 0], [S, 0], [S, S], [0, S]]
        else:
            poly = [vor.vertices[k].tolist() for k in reg]
        poly = clip_rect(poly, S)
        cells.append(np.asarray(poly, float))
    return cells


def make_cells(n, S, seed, lloyd=1):
    rng = np.random.default_rng(seed)
    seeds = rng.random((n, 2)) * S
    for _ in range(lloyd):
        cells = voronoi_cells(seeds, S)
        seeds = np.array([poly_centroid(c) if len(c) >= 3 else rng.random(2) * S for c in cells])
    cells = voronoi_cells(seeds, S)
    return [c for c in cells if len(c) >= 3 and poly_area(c) > 8]


# ---------------------------------------------------------------- 散开
def scatter(cells, out_side, seed, gap=8.0):
    rng = np.random.default_rng(seed + 999)
    order = sorted(range(len(cells)), key=lambda i: -poly_area(cells[i]))
    placed_polys, polys = [], [None] * len(cells)
    poses = [None] * len(cells)          # 按片号对齐存位姿
    for idx in order:
        cell = cells[idx]
        cen = poly_centroid(cell)
        rad = np.max(np.linalg.norm(cell - cen, axis=1))
        for _t in range(4000):
            theta = rng.uniform(0, 2 * np.pi)
            cx = rad + 4 + rng.random() * (out_side - 2 * (rad + 4))
            cy = rad + 4 + rng.random() * (out_side - 2 * (rad + 4))
            p = transform(cell - cen, rot(theta), np.array([cx, cy]))
            if p[:, 0].min() < 0 or p[:, 0].max() > out_side or p[:, 1].min() < 0 or p[:, 1].max() > out_side:
                continue
            if any(convex_overlap(p, q, gap) for q in placed_polys):
                continue
            placed_polys.append(p)
            poses[idx] = (theta, np.array([cx, cy]), cen)
            polys[idx] = p
            break
        else:
            raise RuntimeError("散开摆放失败（画布太小）")
    return polys, poses


# ---------------------------------------------------------------- 求解
def _edges(poly):
    out = []
    n = len(poly)
    for k in range(n):
        p0 = poly[k]; p1 = poly[(k + 1) % n]
        out.append((k, float(np.linalg.norm(p1 - p0))))
    return out


def _rigid_fit(q0, q1, P0, P1):
    """刚体变换(纯旋转+平移)把 (q0,q1) 映射到 (P0,P1)。返回 (R,t)。"""
    ang = np.arctan2(P1[1] - P0[1], P1[0] - P0[0]) - np.arctan2(q1[1] - q0[1], q1[0] - q0[0])
    R = rot(ang)
    t = P0 - R @ q0
    return R, t


def _samp(img, xy):
    x, y = float(xy[0]), float(xy[1])
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < img.shape[0] and 0 <= xi < img.shape[1]:
        return img[yi, xi].astype(float)
    return None


def _shape_resid(a0, a1, b0, b1, K=12):
    """两条撕痕边的形状/曲率一致性：把 B 边刚性贴到 A 边，求均残差。
    直边(合成 Voronoi 边)恒 ~0；手撕曲边/锯齿边靠这个区分形状对不对。"""
    A = np.linspace(a0, a1, K); B = np.linspace(b0, b1, K)
    R, t = _rigid_fit(b0, b1, a0, a1)
    return float(np.linalg.norm(A - (B @ R.T + t), axis=1).mean())


def _tex_continuity(scat, a0, a1, cenA, poseA, poseB, d=4.0, K=20):
    """纹理/内容连续：拼好后接缝两侧各向内取 d 像素，比较画面颜色是否连续。
    内容能对上=拼对(差值小)；接错边=两边内容错位(差值大)。"""
    RA, tA = poseA; RB, tB = poseB
    e = a1 - a0; L = float(np.linalg.norm(e))
    if L < 1e-6:
        return 0.0
    nrm = np.array([-e[1], e[0]]) / L
    mid = (a0 + a1) / 2
    if np.dot(nrm, cenA - mid) < 0:
        nrm = -nrm
    diffs = []
    for s in np.linspace(0.15, 0.85, K):
        p = a0 + e * s
        ca = _samp(scat, RA.T @ (p + nrm * d - tA))   # A 内侧像
        cb = _samp(scat, RB.T @ (p - nrm * d - tB))   # 接缝另一侧(B)像
        if ca is None or cb is None:
            continue
        diffs.append(float(np.abs(ca - cb).mean()))
    return float(np.mean(diffs)) if diffs else 0.0


def solve(polys, len_tol=1e-6, scat=None, shape_tol=6.0, tex_tol=75.0):
    n = len(polys)
    # 统一成 CCW，避免绕向不一致时“反向贴边”贴错面
    polys = [p if _signed_area(p) > 0 else p[::-1] for p in polys]
    # 所有边（全局），按长度分桶
    all_edges = []  # (pid, k, length, p0, p1)
    for pid, poly in enumerate(polys):
        m = len(poly)
        for k in range(m):
            p0 = poly[k]; p1 = poly[(k + 1) % m]
            all_edges.append((pid, k, float(np.linalg.norm(p1 - p0)), p0, p1))
    # 长度 -> 边索引
    by_len = [e[2] for e in all_edges]
    order = np.argsort(by_len)
    left = 0
    buckets = []
    for oi in range(len(order)):
        buckets.append(order[oi])
    # 找每条边的「等长对边」候选
    def candidates(ei):
        L = all_edges[ei][2]; pid = all_edges[ei][0]
        res = []
        for ej in range(len(all_edges)):
            if ej == ei: continue
            e2 = all_edges[ej]
            if e2[0] == pid: continue
            if abs(e2[2] - L) <= len_tol:
                res.append(ej)
        res.sort(key=lambda ej: abs(all_edges[ej][2] - L))
        return res

    # BFS 逐片长出
    pose = [None] * n
    pose[0] = (np.eye(2), np.zeros(2))
    placed_polys = {0: polys[0].copy()}
    from collections import deque
    q = deque([0])
    shape_rej = 0; tex_rej = 0; tex_vals = []
    while q:
        A = q.popleft()
        RA, tA = pose[A]
        polyA = placed_polys[A]
        # A 的每条边
        degA = len(polys[A])
        for ka in range(degA):
            ea0 = polyA[ka]; ea1 = polyA[(ka + 1) % degA]
            # 在原始边表里找 A 的这条边（用变换前的长度定位）——直接用长度匹配候选即可
            La = float(np.linalg.norm(ea1 - ea0))
            # 候选：其它未放片里、某条边长度≈La 的
            for ej in range(len(all_edges)):
                B, kb, Lb, qb0, qb1 = all_edges[ej]
                if B == A or pose[B] is not None: continue
                if abs(Lb - La) > len_tol: continue
                # 贴边：两种端点对应都试，取“与已放片不重叠”的那种（绕向鲁棒）
                got = None
                for P0, P1 in ((ea1, ea0), (ea0, ea1)):
                    Rb, tb = _rigid_fit(qb0, qb1, P0, P1)
                    pB = transform(polys[B], Rb, tb)
                    if any(convex_overlap(pB, P2, -0.2) for P2 in placed_polys.values()):  # -0.2 容忍共享边浮点
                        continue
                    # ① 形状/曲率一致性：两条撕痕边的形状须贴合（直边恒成立，曲边/手撕靠这个）
                    if _shape_resid(ea0, ea1, qb0, qb1) > shape_tol:
                        shape_rej += 1; continue
                    # ② 纹理/内容连续：拼好后接缝两侧画面颜色要连续（接错边=内容错位=差值大）
                    if scat is not None:
                        td = _tex_continuity(scat, ea0, ea1, placed_polys[A].mean(0), pose[A], (Rb, tb))
                        tex_vals.append(td)
                        if td > tex_tol:
                            tex_rej += 1; continue
                    got = (Rb, tb, pB); break
                if got is None:
                    continue
                Rb, tb, pB = got
                pose[B] = (Rb, tb); placed_polys[B] = pB; q.append(B)
                break
    placed = sum(1 for p in pose if p is not None)
    print(f"[检测] 形状不通过 {shape_rej} 次, 纹理不连续拒绝 {tex_rej} 次, "
          f"接缝平均纹理差 {round(float(np.mean(tex_vals)),1) if tex_vals else '-'}")
    return pose, placed, placed_polys


# ---------------------------------------------------------------- 校验 & 渲染
def evaluate(placed_polys, S):
    if _HAS_SHAPELY and len(placed_polys) > 1:
        polys = [SPoly(v) for v in placed_polys.values() if len(v) >= 3]
        total = sum(p.area for p in polys)
        u = unary_union(polys)
        overlap = total - u.area
        # 外接面积 vs 并集面积 → 空洞/外溢近似：用凸包面积减并集
        hull = u.convex_hull.area
        return {"union_area": float(u.area), "sum_area": float(total),
                "overlap_area": float(max(0.0, overlap)),
                "overlap_pct": float(max(0.0, overlap) / max(total, 1e-9) * 100),
                "hull_frac": float(u.area / max(hull, 1e-9))}
    return {}


def render_assembled(scat, scat_side, polys, pose, out_side):
    """用散落图 + 求解位姿复原（只依赖 solver 输出，验证真实重建）。"""
    import cv2
    canvas = np.zeros((out_side, out_side, 3), np.uint8)
    for pid, poly in enumerate(polys):
        if pose[pid] is None: continue
        R, t = pose[pid]
        p = transform(poly, R, t)
        x0 = int(np.floor(p[:, 0].min())) - 2; y0 = int(np.floor(p[:, 1].min())) - 2
        x1 = int(np.ceil(p[:, 0].max())) + 2; y1 = int(np.ceil(p[:, 1].max())) + 2
        if x0 < 0 or y0 < 0 or x1 > out_side or y1 > out_side:  # 超界则跳过该片渲染
            continue
        # 源：从散落图裁剪该片
        sx0 = max(0, int(poly[:, 0].min()) - 2); sy0 = max(0, int(poly[:, 1].min()) - 2)
        sx1 = min(scat_side, int(poly[:, 0].max()) + 2); sy1 = min(scat_side, int(poly[:, 1].max()) + 2)
        crop = scat[sy0:sy1, sx0:sx1]
        mask = np.zeros((sy1 - sy0, sx1 - sx0), np.uint8)
        cv2.fillPoly(mask, [np.round(poly - [sx0, sy0]).astype(np.int32)], 255)
        # warp 源(散落坐标) → 复原坐标
        # 源裁剪以 (sx0,sy0) 为原点：assembled = R*(sx0+u, sy0+v)+t = R*(u,v) + (R*(sx0,sy0)+t)
        shift = R @ np.array([sx0, sy0], float)
        M = np.array([[R[0, 0], R[0, 1], t[0] + shift[0]],
                      [R[1, 0], R[1, 1], t[1] + shift[1]]], float)
        warped = cv2.warpAffine(crop, M, (out_side, out_side), flags=cv2.INTER_LINEAR)
        wmask = cv2.warpAffine(mask, M, (out_side, out_side), flags=cv2.INTER_NEAREST)
        m3 = (wmask > 0)[..., None]
        canvas = np.where(m3, warped, canvas)
    return canvas


def render_scatter(scat, scat_side, polys):
    import cv2
    cv2.imwrite(str(TMP_SCAT), scat)


# ---------------------------------------------------------------- main
TMP_SCAT = pathlib.Path("tmp_scat.png")


def read_color(p):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def write_color(p, im):
    import cv2
    ok, enc = cv2.imencode(pathlib.Path(p).suffix or ".png", im)
    enc.tofile(str(p))


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--image", default="拼图还原项目/svg_tool/test_out/letter_h5.png")
    ap.add_argument("--out", default="拼图还原项目/svg_tool/test_out/poly_solve")
    ap.add_argument("--len-tol", type=float, default=1e-6)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    tex = read_color(args.image)
    if tex is None:
        print("读不到纹理:", args.image); return 1
    S = tex.shape[0]

    t0 = time.time()
    cells = make_cells(args.n, S, args.seed)
    t_cells = time.time() - t0

    scat_side = int(S * 1.8)
    t0 = time.time()
    polys, poses = scatter(cells, scat_side, args.seed)
    t_scatter = time.time() - t0

    # 渲染散落图（用纹理按散落位姿贴）
    scat = np.zeros((scat_side, scat_side, 3), np.uint8)
    for cell, p in zip(cells, polys):
        # cell(letter) -> scatter: 由 poses 存的是散落时的 (theta, c, cen)
        pass
    # poses[idx]=(theta, center, cen)；polys[idx] 已是散落多边形。重建映射：
    # 散落多边形 p 与 cell 的关系: p = R(theta)*(cell-cen)+center
    for idx, (theta, center, cen) in enumerate(poses):
        cell = cells[idx]
        R = rot(theta)
        # cell -> scatter
        M = np.array([[R[0, 0], R[0, 1], center[0] - (R @ cen)[0]],
                      [R[1, 0], R[1, 1], center[1] - (R @ cen)[1]]], float)
        warped = cv2.warpAffine(tex, M, (scat_side, scat_side), flags=cv2.INTER_LINEAR)
        mask = np.zeros((S, S), np.uint8)
        cv2.fillPoly(mask, [np.round(cell).astype(np.int32)], 255)
        wmask = cv2.warpAffine(mask, M, (scat_side, scat_side), flags=cv2.INTER_NEAREST)
        m3 = (wmask > 0)[..., None]
        scat = np.where(m3, warped, scat)

    t0 = time.time()
    pose, placed, placed_polys = solve(polys, args.len_tol, scat=scat)
    t_solve = time.time() - t0

    metrics = evaluate(placed_polys, S)

    # 复原画布尺寸 = 已放碎片的 bbox
    allp = np.vstack([placed_polys[k] for k in placed_polys])
    x0 = int(np.floor(allp[:, 0].min())) - 2; y0 = int(np.floor(allp[:, 1].min())) - 2
    x1 = int(np.ceil(allp[:, 0].max())) + 2; y1 = int(np.ceil(allp[:, 1].max())) + 2
    shift = np.array([-x0, -y0], float)
    pose_shifted = []
    for ps in pose:
        if ps is None: pose_shifted.append(None)
        else:
            R, t = ps; pose_shifted.append((R, t + shift))
    out_side = max(x1 - x0, y1 - y0)
    t0 = time.time()
    if placed >= 2:
        asm = render_assembled(scat, scat_side, polys, pose_shifted, out_side)
    else:
        asm = np.zeros((out_side, out_side, 3), np.uint8)
    t_render = time.time() - t0

    write_color(outdir / "scattered.png", scat)
    write_color(outdir / "assembled.png", asm)

    report = {
        "n_requested": args.n, "pieces": len(polys), "placed": placed,
        "metrics": metrics,
        "time_s": {"cells": round(t_cells, 2), "scatter": round(t_scatter, 2),
                   "solve": round(t_solve, 3), "render": round(t_render, 2),
                   "total": round(t_cells + t_scatter + t_solve + t_render, 2)},
        "assembled_size": [out_side, out_side],
    }
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if placed == len(polys) else 2


if __name__ == "__main__":
    raise SystemExit(main())
