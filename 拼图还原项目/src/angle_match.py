"""角度匹配：凹形多边形的夹角 vs 三角形碎片的内角（允许误差），嵌入渲染拼大图。

思路
----
* 每块非凹碎片先「三角形化」：取其最大面积的内接三角形（顶点取多边形顶点组合），
  得到 3 个主顶点与 3 个内角。照片上碎片是三角形，但拟合时角被弧边分裂成 4~5 个点；
* 凹形多边形每个角位 i（由两条邻边构成）的夹角 φ_i，与三角形每个内角 ψ_v 做容差匹配
  |φ_i − ψ_v| ≤ tol_deg；
* 匹配成功即「三角形的这只角可以嵌进凹形的那只角」，用 3 锚点相似拟合
  （三角形顶点 v + 两条邻边端点 ↔ 凹形角 i + 两条邻边端点）算出旋转/缩放/平移，
  把整个三角形放进去，并统计与凹形的重叠比例（穿透）作为质量标注；
* 每格画一个匹配（凹形 + 嵌入三角形），全部拼到一张大图。

用法:
    python src/angle_match.py --pieces out5/pieces_scaled.json --out out5 --tol-deg 6 --top 6
输出:
    out5/40_angle_match.png
    out5/40_angle_match.json
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
import geometry as G  # noqa: E402

PALETTE = [(70, 70, 200), (40, 120, 230), (40, 150, 60), (200, 110, 30), (160, 60, 160), (30, 150, 170)]


# ---------------------------------------------------------------- 几何工具
def best_triangle(poly: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """取多边形上面积最大的内接三角形（顶点取多边形顶点组合），返回 (三角形顶点, 三个内角)。"""
    n = len(poly)
    best, bi, bj, bk = -1.0, 0, 1, 2
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = float(abs(cv2.contourArea(
                    poly[[i, j, k]].astype(np.float32).reshape(-1, 1, 2))))
                if a > best:
                    best, bi, bj, bk = a, i, j, k
    tri = poly[[bi, bj, bk]]
    angs = G.interior_angles(tri)
    return tri, angs


def similarity_fit(src, dst):
    """最小二乘相似变换 p' = s*R*p + t（det R = +1，即不含镜像）。"""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = ((src - mu_s).T @ (dst - mu_d)) / len(src)
    U, D, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    var_s = ((src - mu_s) ** 2).sum() / len(src)
    s = float(np.sum(D) / var_s) if var_s > 0 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def apply_fit(poly, s, R, t):
    return (s * (np.asarray(poly, float) @ R.T)) + t


def convex_area(poly):
    return abs(float(cv2.contourArea(cv2.convexHull(np.asarray(poly, np.float32).reshape(-1, 1, 2)))))


# ---------------------------------------------------------------- 渲染
def tile(pieces, concave_id, tri_id, poly_n, poly_t, title, sub, w=520, h=520):
    ids = [concave_id, tri_id]
    allpts = np.vstack([np.asarray(poly_n, float), np.asarray(poly_t, float)])
    lo, hi = allpts.min(0), allpts.max(0)
    span = max(float((hi - lo).max()), 1.0)
    k = min(w, h) * 0.80 / span
    ox = (w - (hi[0] - lo[0]) * k) / 2 - lo[0] * k
    oy = (h - (hi[1] - lo[1]) * k) / 2 - lo[1] * k

    def to_px(p):
        p = np.asarray(p, float).copy()
        p[:, 0] = p[:, 0] * k + ox
        p[:, 1] = p[:, 1] * k + oy
        return p.astype(np.int32)

    cv = np.full((h + 56, w, 3), 255, np.uint8)
    # 凹形：浅色底 + 深描边
    poly_n_px = to_px(poly_n)
    fill_n = tuple(int(c) for c in PALETTE[concave_id % len(PALETTE)])
    fill_n = (fill_n[0] // 2 + 120, fill_n[1] // 2 + 120, fill_n[2] // 2 + 120)
    cv2.fillPoly(cv, [poly_n_px], fill_n)
    cv2.polylines(cv, [poly_n_px], True, (40, 40, 40), 2, cv2.LINE_AA)
    c = poly_n_px.mean(0).astype(int)
    cv2.putText(cv, f"#{concave_id}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # 嵌入三角形：实色 + 白描边
    poly_t_px = to_px(poly_t)
    cv2.fillPoly(cv, [poly_t_px], PALETTE[tri_id % len(PALETTE)])
    cv2.polylines(cv, [poly_t_px], True, (255, 255, 255), 2, cv2.LINE_AA)
    c = poly_t_px.mean(0).astype(int)
    cv2.putText(cv, f"#{tri_id}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(cv, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(cv, sub, (8, h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
    return cv


def text_panel(lines, w=520, h=576):
    cv = np.full((h, w, 3), 255, np.uint8)
    y = 26
    for ln, bold in lines:
        cv2.putText(cv, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 if not bold else 0.5,
                    (20, 20, 20) if bold else (90, 90, 90), 1, cv2.LINE_AA)
        y += 23
    return cv


def overview_tile(pieces, concave_id, poly_n, w=520, h=520):
    """总览格：凹形 + 所有三角形（原始位置）。"""
    tris = [np.asarray(pieces[pid]["polygon"], float)
            for pid in sorted(pieces) if pid != concave_id]
    allpts = np.vstack([poly_n] + tris)
    lo, hi = allpts.min(0), allpts.max(0)
    span = max(float((hi - lo).max()), 1.0)
    k = min(w, h) * 0.80 / span
    ox = (w - (hi[0] - lo[0]) * k) / 2 - lo[0] * k
    oy = (h - (hi[1] - lo[1]) * k) / 2 - lo[1] * k

    def to_px(p):
        p = np.asarray(p, float).copy()
        p[:, 0] = p[:, 0] * k + ox
        p[:, 1] = p[:, 1] * k + oy
        return p.astype(np.int32)

    cv = np.full((h + 56, w, 3), 255, np.uint8)
    poly_n_px = to_px(poly_n)
    fill_n = tuple(int(c) for c in PALETTE[concave_id % len(PALETTE)])
    fill_n = (fill_n[0] // 2 + 120, fill_n[1] // 2 + 120, fill_n[2] // 2 + 120)
    cv2.fillPoly(cv, [poly_n_px], fill_n)
    cv2.polylines(cv, [poly_n_px], True, (40, 40, 40), 2, cv2.LINE_AA)
    c = poly_n_px.mean(0).astype(int)
    cv2.putText(cv, f"#{concave_id}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    for pid in sorted(pieces):
        if pid == concave_id:
            continue
        poly_px = to_px(np.asarray(pieces[pid]["polygon"], float))
        cv2.fillPoly(cv, [poly_px], PALETTE[pid % len(PALETTE)])
        cv2.polylines(cv, [poly_px], True, (255, 255, 255), 2, cv2.LINE_AA)
        c = poly_px.mean(0).astype(int)
        cv2.putText(cv, f"#{pid}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(cv, f"ALL PIECES (concave #{concave_id} + triangles)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(cv, f"fill {pieces[concave_id]['_fill']*100:.1f}% of convex hull", (8, h + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
    return cv


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out5/pieces_scaled.json")
    ap.add_argument("--out", default="out5")
    ap.add_argument("--tol-deg", type=float, default=6.0, help="角度容差（度）")
    ap.add_argument("--max-ov", type=float, default=0.35, help="嵌入后与凹形重叠比例上限（穿透过滤）")
    ap.add_argument("--top", type=int, default=6, help="每块三角形最多画几个匹配")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = {int(d["id"]): d for d in data["pieces"]}

    # 找凹形：真实面积明显小于凸包面积的那块
    concave = None
    for pid, d in pieces.items():
        poly = np.array(d["polygon"], float)
        fill = G.polygon_area(poly) / max(convex_area(poly), 1e-9)
        d["_fill"] = fill
        if fill < 0.9:
            concave = pid
    if concave is None:
        concave = max(pieces, key=lambda p: pieces[p]["area"])
    print(f"凹形碎片：#{concave}（面积占凸包 {pieces[concave]['_fill']*100:.1f}%）")

    aim = pieces[concave]
    Npoly = np.array(aim["polygon"], float)
    Nedges = aim["edges"]
    n = len(Nedges)
    phi = aim["interior_angles"]

    # 各碎片三角形化
    tri_info = {}
    print("\n三角形化（最大面积内接三角形）:")
    for pid in sorted(pieces):
        if pid == concave:
            continue
        tri, angs = best_triangle(np.array(pieces[pid]["polygon"], float))
        tri_info[pid] = {"tri": tri, "angs": angs}
        print(f"  #{pid}: 三角形内角 {[round(a, 1) for a in angs]}  "
              f"(原拟合 {len(pieces[pid]['polygon'])} 顶点)")

    print(f"\n凹形 #{concave} 各夹角: " + "  ".join(f"{round(a,1)}°" for a in phi))

    # 角度匹配
    results = []
    for tid, info in tri_info.items():
        Tpoly, Tangs = info["tri"], info["angs"]
        m = len(Tangs)
        for i in range(n):                     # 凹形角位 (i, i+1)
            e1, e2 = Nedges[i], Nedges[(i + 1) % n]
            for v in range(m):                 # 三角形角 v
                dphi = abs(phi[i] - Tangs[v])
                if dphi > args.tol_deg:
                    continue
                # 锚点：三角形 (v-1, v, v+1) ↔ 凹形 (i, i+1, i+2)
                vv = [(v - 1) % m, v, (v + 1) % m]
                src = Tpoly[vv]
                P = [np.array(Nedges[i]["a"], float), np.array(Nedges[i]["b"], float),
                     np.array(Nedges[(i + 1) % n]["b"], float)]
                dst = np.array([P[0], P[1], P[2]], float)
                s, R, t = similarity_fit(src, dst)
                placed = apply_fit(Tpoly, s, R, t)
                ov = G.poly_overlap_area(placed, Npoly) / max(G.polygon_area(Tpoly), 1e-9)
                if ov > args.max_ov:
                    continue
                results.append(dict(tid=tid, i=i, v=v, dphi=dphi,
                                    phi_i=phi[i], psi_v=float(Tangs[v]),
                                    scale=s, ov=ov, placed=placed))

    results.sort(key=lambda r: (r["tid"], r["dphi"]))
    print(f"\n角度吻合（|Δ|≤{args.tol_deg:.0f}°，穿透≤{args.max_ov*100:.0f}%）的角位：{len(results)} 个")
    for r in results[: 24]:
        print(f"  #{r['tid']} 角{r['v']}  →  #{concave} 角位 {r['i']}: "
              f"{r['psi_v']:.1f}° vs {r['phi_i']:.1f}°  Δ{r['dphi']:.1f}°  "
              f"缩放 ×{r['scale']:.3f}  穿透 {r['ov']*100:.1f}%")

    # 渲染
    tiles = []
    lines = [("ANGLE MATCHING (concave corners vs triangle corners)", True), ("", False)]
    lines.append((f"concave #{concave}: {n} corners", True))
    for a in phi:
        lines.append((f"     {a:6.1f}°", False))
    for pid in sorted(tri_info):
        lines.append((f"#{pid} triangle:  {', '.join(f'{a:.1f}°' for a in tri_info[pid]['angs'])}", True))
    lines.append((f"tolerance ±{args.tol_deg:.0f}°   max penetration {args.max_ov*100:.0f}%", False))
    tiles.append(text_panel(lines))
    tiles.append(overview_tile(pieces, concave, Npoly))

    for tid in sorted(tri_info):
        sub = [r for r in results if r["tid"] == tid][: args.top]
        for r in sub:
            t = tile(pieces, concave, tid, Npoly, r["placed"],
                     f"#{tid} corner {r['v']}  ->  #{concave} corner {r['i']}",
                     f"{r['psi_v']:.1f}° vs {r['phi_i']:.1f}°   Δ{r['dphi']:.1f}°   "
                     f"scale ×{r['scale']:.3f}   pen {r['ov']*100:.1f}%")
            tiles.append(t)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].shape[1], tiles[0].shape[0]
    big = np.full((64 + rows * th + 60, cols * tw + 20, 3), 255, np.uint8)
    cv2.putText(big, f"angle matching - {len(results)} corner fits (tol ±{args.tol_deg:.0f}°, no mirror)",
                (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        big[64 + r * th:64 + r * th + t.shape[0], 10 + c * tw:10 + c * tw + t.shape[1]] = t
    out_png = outdir / "40_angle_match.png"
    cv2.imwrite(str(out_png), big)
    print(f"写出 {out_png}  {big.shape[1]}x{big.shape[0]}")

    (outdir / "40_angle_match.json").write_text(
        json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in r.items()} for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
