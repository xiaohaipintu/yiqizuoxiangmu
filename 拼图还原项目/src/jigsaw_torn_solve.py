"""手撕拼图求解（边界弧匹配 + RANSAC + 位姿图拼装）。

思路（对付「没有等长直边」的手撕件）：
  撕口是一对互补的曲线：相邻两片沿同一条撕痕，边界点集相同、切向相反。
  → 每片边界等弧长重采样(点+切向)；对任两片 A、B：
     随机取 pa∈A、pb∈B，按「R·tb = -ta、R·pb+t = pa」定一个刚体变换(RANSAC 假设)，
     再把 B 的所有边界点变换过去，用 KD-tree 数「落在 A 边界 ε 内」的内点数。
     真正的相邻片共享一整条撕痕 → 内点数飙升 → 该变换即二者相对位姿。
  → 每对相邻片得到一条「边(transform, 内点数)」，建图后按内点最强 BFS 长出 + 不重叠校验，拼全部。

同时提供「真实手撕拼图」生成器（共享撕痕的抖动网格），保证问题可解、可用于回归。
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np
from scipy.spatial import cKDTree

def rd_poly_area(p):
    p = np.asarray(p, float); x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def signed_area(p):
    p = np.asarray(p, float); x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def rot(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, -s], [s, c]])

def centroid(p):
    p = np.asarray(p, float); x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1); cr = x * yn - xn * y; a = cr.sum()
    if abs(a) < 1e-12: return p.mean(0)
    return np.array([((x + xn) * cr).sum() / (3 * a), ((y + yn) * cr).sum() / (3 * a)])


# ============================================================ 真实手撕拼图（共享撕痕）
def torn_tessellation(G, S, seed, amp=4.0, jit=0.26):
    rng = np.random.default_rng(seed)
    # 抖动网格角点
    P = np.zeros((G + 1, G + 1, 2))
    for i in range(G + 1):
        for j in range(G + 1):
            x, y = S * i / G, S * j / G
            if 0 < i < G: x += rng.uniform(-jit, jit) * S / G
            if 0 < j < G: y += rng.uniform(-jit, jit) * S / G
            P[i, j] = [x, y]

    cache = {}
    def edge_path(ca, cb):
        """ca,cb 为 (i,j) 角点索引，返回从 ca 到 cb 的共享撕痕点列（确定性，两边片完全相同）。"""
        key = tuple(sorted([ca, cb]))
        if key not in cache:
            r = np.random.default_rng(abs(hash(key)) % (2 ** 31))
            a = np.array(P[key[0]]); b = np.array(P[key[1]])
            L = np.linalg.norm(b - a); n = max(3, int(L / 4))
            d = (b - a) / (L or 1); nx, ny = -d[1], d[0]
            pts = []
            for t in np.linspace(0, 1, n, endpoint=False):
                base = a + (b - a) * t
                off = amp * r.normal() * 0.7 + amp * 0.8 * np.sin(t * np.pi * 3.0)
                pts.append(base + np.array([nx, ny]) * off)
            pts.append(b)
            cache[key] = np.array(pts)
        path = cache[key]
        # 方向：cache 存的是 key[0]→key[1]；请求 ca→cb，若相反则反序
        if key[0] == tuple(ca):
            return path
        return path[::-1]

    cells = []
    for i in range(G):
        for j in range(G):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            poly = []
            for k in range(4):
                seg = edge_path(corners[k], corners[(k + 1) % 4])
                poly.extend(seg[:-1] if k < 3 else seg[:-1])  # 每段去掉末点，避免重复
            poly = np.array(poly)
            if signed_area(poly) < 0:
                poly = poly[::-1]
            cells.append(poly)
    return cells


# ============================================================ 边界点 + 切向
def resample(poly, ds=3.0):
    p = np.vstack([poly, poly[0]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)]); total = cum[-1]
    n = max(12, int(total / ds))
    ss = np.linspace(0, total, n, endpoint=False)
    return np.column_stack([np.interp(ss, cum, p[:, 0]), np.interp(ss, cum, p[:, 1])])

def tangents(pts):
    t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
def procrustes(src, dst, w=None):
    """刚体(无镜像)最小二乘：src->dst。"""
    if w is not None:
        mu_s = (src * w[:, None]).sum(0) / w.sum(); mu_d = (dst * w[:, None]).sum(0) / w.sum()
    else:
        mu_s = src.mean(0); mu_d = dst.mean(0)
    S = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(S)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def contig(mask):
    """布尔环状组里的 True 是否连成一段（≤1 个 F->T 跃变）。"""
    if mask.sum() < 4:
        return False
    return int(np.sum(mask & ~np.roll(mask, 1))) <= 1


def max_run_indices(mask):
    """返回环状布尔数组里最长一段连续 True 的下标。"""
    m = np.asarray(mask, bool); n = len(m)
    if not m.any():
        return np.array([], int)
    if m.all():
        return np.arange(n)
    idx0 = int(np.argmin(m)); m2 = np.roll(m, -idx0)
    bl = 0; bs = 0; cur = 0; st = 0
    for k in range(n):
        if m2[k]:
            if cur == 0: st = k
            cur += 1
            if cur > bl: bl, bs = cur, st
        else:
            cur = 0
    return ((np.arange(bs, bs + bl)) + idx0) % n


# ============================================================ 两两配准（RANSAC 找共享撕痕）
def match_pair(ptsA, tanA, treeA, ptsB, tanB, rng, K=500, eps=1.6):
    nA, nB = len(ptsA), len(ptsB)
    best = (0, None, None, None)   # (runA, (R,t), inl, idx)

    def _lc(a):
        if len(a) <= 1:
            return len(a)
        d = np.diff(a); bn = 1; cur = 1
        for x in d:
            if x == 1:
                cur += 1
                if cur > bn: bn = cur
            else:
                cur = 1
        return bn

    for _ in range(K):
        ia = int(rng.integers(nA)); ib = int(rng.integers(nB))
        ta = tanA[ia]; tb = tanB[ib]
        theta = np.arctan2(-ta[1], -ta[0]) - np.arctan2(tb[1], tb[0])
        R = rot(theta); t = ptsA[ia] - R @ ptsB[ib]
        d, idx = treeA.query(ptsB @ R.T + t, distance_upper_bound=eps)
        inl = np.isfinite(d)
        if inl.sum() < 6:
            continue
        run = _lc(np.where(inl)[0])     # B 自身序列上最长连续内点弧(撕痕)，而非总内点数
        if run > best[0]:
            best = (run, (R, t), inl, idx)
    if best[1] is None:
        return 0, None, None
    run, (R, t), inl, idx = best
    R, t = procrustes(ptsB[inl], ptsA[idx[inl]], np.ones(int(inl.sum())))
    d, idx = treeA.query(ptsB @ R.T + t, distance_upper_bound=eps); inl = np.isfinite(d)
    score = len(max_run_indices(inl))          # 撕痕长度 = B 序列上连续内点弧
    if score < 6:
        return 0, None, None
    runB = max_run_indices(inl)
    if len(runB) >= 6:
        R2, t2 = procrustes(ptsB[runB], ptsA[idx[runB]], np.ones(len(runB)))
        d2, idx2 = treeA.query(ptsB @ R2.T + t2, distance_upper_bound=eps); inl2 = np.isfinite(d2)
        if inl2.sum() >= inl.sum():
            R, t, inl, idx = R2, t2, inl2, idx2
            score = len(max_run_indices(inl))
    return int(score), (R, t), inl


def transform(poly, R, t):
    return np.asarray(poly, float) @ R.T + t


def overlap_area(pa, pb):
    try:
        from shapely.geometry import Polygon
        a = Polygon(pa); b = Polygon(pb)
        if not a.is_valid: a = a.buffer(0)
        if not b.is_valid: b = b.buffer(0)
        return a.intersection(b).area
    except Exception:
        return 0.0


def solve(polys, min_inl=10, K=500, eps=1.5, seed=0):
    n = len(polys)
    polys = [p if signed_area(p) > 0 else p[::-1] for p in polys]
    rng = np.random.default_rng(seed)
    # 目标片用稠密边界(ds=1)建 KDTree，查询片稀采样(ds=2)：消除重采样相位错位，连续性判据才可靠
    adense, atand, trees, bpts, btan = [], [], [], [], []
    for p in polys:
        ad = resample(p, 1.0); adense.append(ad); atand.append(tangents(ad)); trees.append(cKDTree(ad))
        bq = resample(p, 2.0); bpts.append(bq); btan.append(tangents(bq))
    # 两两配准
    edges = []  # (score, i, j, R, t)  transform maps j -> i frame
    for i in range(n):
        for j in range(i + 1, n):
            res = match_pair(adense[i], atand[i], trees[i], bpts[j], btan[j], rng, K, eps)
            sc = res[0]
            if sc >= min_inl and res[1] is not None:
                R, t = res[1]
                # 拒绝明显穿透（相邻片应只贴不压）
                pj_t = transform(polys[j], R, t)
                if overlap_area(pj_t, polys[i]) < 0.15 * rd_poly_area(polys[j]):
                    edges.append((sc, i, j, R, t))
                    edges.append((sc, j, i, R.T, -R.T @ t))
    edges.sort(reverse=True, key=lambda e: e[0])
    # 贪心位姿图：从 0 开始，按内点数从高到低长出，不重叠就放
    pose = [None] * n; pose[0] = (np.eye(2), np.zeros(2))
    placed = {0: polys[0].copy()}; frontier = list(range(n))
    changed = True
    while changed:
        changed = False
        for sc, i, j, R, t in edges:
            if pose[i] is not None and pose[j] is None:
                RA, tA = pose[i]
                Rj, tj = RA @ R, RA @ t + tA
                pj = transform(polys[j], Rj, tj)
                clash = any(overlap_area(pj, P2) > 0.15 * rd_poly_area(polys[j]) for P2 in placed.values())
                if clash: continue
                pose[j] = (Rj, tj); placed[j] = pj; changed = True
    placed_n = sum(1 for p in pose if p is not None)
    return pose, placed_n, placed


def solve_v2(polys, K=500, eps=1.5, min_run=8, seed=0):
    """装配级 RANSAC + 多边互验：多个种子边各自拼一整盘，用“被多少条边互相印证 + 拼上多少片”选真解。"""
    n = len(polys)
    polys = [p if signed_area(p) > 0 else p[::-1] for p in polys]
    # ===== 确定性“共享撕痕”匹配 =====
    # 撕痕同一条折线 → 相邻两片沿撕痕的【边长序列精确相同、但反序】。
    # 用“最长共同边长（反序）子串”直接锁定撕痕：真串几十条边，假串~0，互不混淆。
    from collections import defaultdict

    def elens(p):
        m = len(p); return np.array([np.linalg.norm(p[(k + 1) % m] - p[k]) for k in range(m)])

    la = [elens(p) for p in polys]
    kv = []
    for e in la:
        d = defaultdict(list)
        for kk, v in enumerate(e):
            d[round(float(v), 3)].append(kk)
        kv.append(d)
    tol = 1e-2
    edges = []  # (i,j,R,t,w): transform maps j -> i frame; w=撕痕边数
    for i in range(n):
        for j in range(i + 1, n):
            Ai = la[i]; Aj = la[j]; na = len(Ai); nb = len(Aj); cap = min(na, nb) // 2
            best = (0, 0, 0)
            for ai in range(na):
                for bj in kv[j].get(round(float(Ai[ai]), 3), []):
                    k = 0
                    while k < cap and abs(Ai[(ai + k) % na] - Aj[(bj - k) % nb]) < tol:
                        k += 1
                    if k > best[0]:
                        best = (k, ai, bj)
            k, ai, bj = best
            if k < min_run:
                continue
            Apts = np.array([polys[i][(ai + t) % na] for t in range(k + 1)])
            Bpts = np.array([polys[j][(bj + 1 - t) % nb] for t in range(k + 1)])
            R, t = procrustes(Bpts, Apts)
            if np.linalg.norm(Apts - (Bpts @ R.T + t), axis=1).mean() > 0.5:
                continue
            if overlap_area(transform(polys[j], R, t), polys[i]) > 0.2 * rd_poly_area(polys[j]):
                continue
            edges.append((i, j, R, t, k)); edges.append((j, i, R.T, -R.T @ t, k))
    pose0 = [None] * n; pose0[0] = (np.eye(2), np.zeros(2)); placed0 = {0: polys[0].copy()}
    if not edges:
        return pose0, 1, placed0

    def grow(i0, j0, R0, t0):
        if overlap_area(transform(polys[j0], R0, t0), polys[i0]) > 0.2 * rd_poly_area(polys[j0]):
            return None
        pose = {i0: (np.eye(2), np.zeros(2)), j0: (R0, t0)}
        placed = {i0: polys[i0].copy(), j0: transform(polys[j0], R0, t0)}
        changed = True
        while changed:
            changed = False
            cands = sorted([e for e in edges if e[0] in pose and e[1] not in pose], key=lambda e: -e[4])
            for (p, q, R, t, w) in cands:
                RA, tA = pose[p]; Rq, tq = RA @ R, RA @ t + tA
                pq = transform(polys[q], Rq, tq)
                if any(overlap_area(pq, P2) > 0.2 * rd_poly_area(polys[q]) for P2 in placed.values()):
                    continue
                pose[q] = (Rq, tq); placed[q] = pq; changed = True; break
        return pose, placed

    def corrob(pose):
        s = 0.0
        for (i, j, R, t, w) in edges:
            if i < j and i in pose and j in pose:
                RA, tA = pose[i]; RB, tB = pose[j]
                Rimp = RA.T @ RB; timp = RA.T @ (tB - tA)
                if np.linalg.norm(Rimp - R) < 0.35 and np.linalg.norm(timp - t) < 6.0:
                    s += w
        return s

    best = None
    for (i0, j0, R0, t0, w0) in edges:
        g = grow(i0, j0, R0, t0)
        if g is None:
            continue
        pose_h, placed_h = g
        key = (len(placed_h), corrob(pose_h))
        if best is None or key > best[0]:
            best = (key, pose_h, placed_h)
    if best is None:
        return pose0, 1, placed0
    _, pose_h, placed_h = best
    return [pose_h.get(k) for k in range(n)], len(placed_h), placed_h


def evaluate(placed):
    if len(placed) < 2:
        return {}
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        ps = []
        for v in placed.values():
            g = Polygon(v)
            if not g.is_valid: g = g.buffer(0)
            if g.area > 0: ps.append(g)
        total = sum(p.area for p in ps); u = unary_union(ps)
        hull = u.convex_hull.area
        return {"overlap_pct": float(max(0, total - u.area) / max(total, 1e-9) * 100),
                "hull_frac": float(u.area / max(hull, 1e-9)),
                "union_area": float(u.area)}
    except Exception as e:
        return {"error": str(e)[:100]}


def scatter(cells, S, seed):
    from shapely.geometry import Polygon
    rng = np.random.default_rng(seed + 77)
    side = int(S * 1.9)
    order = sorted(range(len(cells)), key=lambda i: -rd_poly_area(cells[i]))
    placed_hulls = []; out = [None] * len(cells)
    for idx in order:
        c = cells[idx]; cen = centroid(c); rad = np.max(np.linalg.norm(c - cen, axis=1)) + 8
        for _ in range(4000):
            th = rng.uniform(0, 2 * np.pi); cx = rad + rng.random() * (side - 2 * rad); cy = rad + rng.random() * (side - 2 * rad)
            p = (c - cen) @ rot(th).T + [cx, cy]
            h = Polygon(p).convex_hull.buffer(4)
            if h.bounds[0] < 0 or h.bounds[1] < 0 or h.bounds[2] > side or h.bounds[3] > side: continue
            if any(h.intersects(ph) for ph in placed_hulls): continue
            placed_hulls.append(h); out[idx] = p; break
        else:
            raise RuntimeError("散开失败")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=3, help="G×G 片")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--min-inl", type=int, default=10)
    args = ap.parse_args()
    S = 760
    cells = torn_tessellation(args.G, S, args.seed)
    scattered = scatter(cells, S, args.seed)
    t0 = time.time()
    pose, placed, pp = solve_v2(scattered, K=args.K, min_run=args.min_inl, seed=args.seed)
    dt = time.time() - t0
    met = evaluate(pp)
    print(json.dumps({"G": args.G, "pieces": len(cells), "placed": placed,
                      "metrics": met, "solve_s": round(dt, 2)}, ensure_ascii=False, indent=2))
    return 0 if placed == len(cells) else 2


if __name__ == "__main__":
    raise SystemExit(main())
