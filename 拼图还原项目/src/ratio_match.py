"""按「边长比例」而不是绝对长度做匹配。

思路
----
每块碎片先算自己的「比例指纹」：各边长度 / 该块周长（尺度无关）。
三角形要嵌进凹形的某个角，就得让三角形的两条邻边与凹形的两条邻边
*比例一致*（L_n1/L_t1 ≈ L_n2/L_t2）。用残差 |log(s1/s2)| 排序，
比例吻合的角就是它能进去的槽；再按拟合出的相似变换（旋转+平移+整体缩放，
不含镜像）放进去，检查有没有穿透。

用法: python src/ratio_match.py --pieces out4/pieces.json --out out4 --top 4
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
def similarity_fit(src, dst):
    """最小二乘相似变换 p' = s*R*p + t（det R = +1，即不含镜像）。"""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = ((src - mu_s).T @ (dst - mu_d)) / len(src)
    U, D, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                       # 强制不镜像
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


def fingerprint(lens):
    p = sum(lens)
    return [L / p for L in lens]


# ---------------------------------------------------------------- 渲染
def tile(pieces, ids, polys_extra, title, sub, w=470, h=470):
    allpts = np.vstack([np.asarray(p, float) for p in polys_extra])
    for pid in ids:
        allpts = np.vstack([allpts, np.asarray(pieces[pid]["polygon"], float)])
    lo, hi = allpts.min(0), allpts.max(0)
    span = max(float((hi - lo).max()), 1.0)
    k = min(w, h) * 0.82 / span
    ox = (w - (hi[0] - lo[0]) * k) / 2 - lo[0] * k
    oy = (h - (hi[1] - lo[1]) * k) / 2 - lo[1] * k

    def to_px(p):
        p = np.asarray(p, float).copy()
        p[:, 0] = p[:, 0] * k + ox
        p[:, 1] = p[:, 1] * k + oy
        return p.astype(np.int32)

    cv = np.full((h + 56, w, 3), 255, np.uint8)
    for pid in ids:
        poly = to_px(pieces[pid]["polygon"])
        cv2.fillPoly(cv, [poly], PALETTE[pid % len(PALETTE)])
        cv2.polylines(cv, [poly], True, (210, 210, 210), 6, cv2.LINE_AA)
        cv2.polylines(cv, [poly], True, (40, 40, 40), 2, cv2.LINE_AA)
        c = poly.mean(0).astype(int)
        cv2.putText(cv, f"#{pid}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(cv, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(cv, sub, (8, h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 110, 110), 1, cv2.LINE_AA)
    return cv


def text_panel(lines, w=470, h=526):
    cv = np.full((h, w, 3), 255, np.uint8)
    y = 26
    for ln, bold in lines:
        cv2.putText(cv, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44 if not bold else 0.5,
                    (20, 20, 20) if bold else (90, 90, 90), 1, cv2.LINE_AA)
        y += 24
    return cv


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out4/pieces.json")
    ap.add_argument("--out", default="out4")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--max-resid", type=float, default=0.25, help="比例残差上限 |log(s1/s2)|")
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

    print("\n各块比例指纹（边长 / 该块周长，尺度无关）:")
    for pid in sorted(pieces):
        d = pieces[pid]
        lens = [e["length"] for e in d["edges"]]
        fp = fingerprint(lens)
        print(f"  #{pid}: 周长 {sum(lens):7.1f}px  " +
              "  ".join(f"{L:5.1f}({f*100:4.1f}%)" for L, f in zip(lens, fp)))

    aim = pieces[concave]
    Npoly = np.array(aim["polygon"], float)
    Nedges = aim["edges"]
    n = len(Nedges)

    results = []
    tris = [p for p in sorted(pieces) if p != concave]
    for tid in tris:
        T = pieces[tid]
        Tpoly = np.array(T["polygon"], float)
        Tlen = [e["length"] for e in T["edges"]]
        m = len(Tlen)
        for i in range(n):                          # N 的相邻两边 (i, i+1)
            e1, e2 = Nedges[i], Nedges[(i + 1) % n]
            Ln1, Ln2 = e1["length"], e2["length"]
            for v in range(m):                      # 三角形对应的共享顶点
                Lt1 = Tlen[(v - 1) % m]
                Lt2 = Tlen[v]
                s1, s2 = Ln1 / Lt1, Ln2 / Lt2
                resid = abs(math.log(s1 / s2))
                if resid > args.max_resid:
                    continue
                # 顶点链：三角形 (v-1, v, v+1) ↔ N (i, i+1, i+2)
                vv = [(v - 1) % m, v, (v + 1) % m]
                src = Tpoly[vv]
                P = [np.array(Nedges[i]["a"], float), np.array(Nedges[i]["b"], float),
                     np.array(Nedges[(i + 1) % n]["b"], float)]
                dst = np.array([P[0], P[1], P[2]], float)
                s, R, t = similarity_fit(src, dst)
                placed = apply_fit(Tpoly, s, R, t)
                ov = G.poly_overlap_area(placed, Npoly) / max(G.polygon_area(Tpoly), 1e-9)
                if ov > 0.03:                       # 明显穿透的不算
                    continue
                results.append(dict(tid=tid, i=i, v=v, resid=resid, scale=s,
                                    s1=s1, s2=s2, ov=ov, placed=placed,
                                    Ln1=Ln1, Ln2=Ln2, Lt1=Lt1, Lt2=Lt2,
                                    seam=abs(s1 - s2) / max(s1, s2)))

    print(f"\n比例吻合（残差≤{args.max_resid*100:.0f}%）且不穿透的角位：{len(results)} 个")

    tiles = []
    # 第一格：全部碎片 + 比例指纹摘要
    lines = [("PROPORTION FINGERPRINT (edge / perimeter)", True), ("", False)]
    for pid in sorted(pieces):
        d = pieces[pid]
        lens = [e["length"] for e in d["edges"]]
        fp = fingerprint(lens)
        lines.append((f"#{pid}  perimeter {sum(lens):.0f}px", True))
        for L, f in zip(lens, fp):
            lines.append((f"     {L:6.1f} px   {f*100:5.1f}%", False))
    tiles.append(text_panel(lines))
    tiles.append(tile(pieces, sorted(pieces), [Npoly], f"ALL {len(pieces)} PIECES",
                      f"concave piece = #{concave}", w=470))

    # 每块三角形按残差从好到差各取 top 个
    for tid in tris:
        sub = sorted([r for r in results if r["tid"] == tid], key=lambda r: r["resid"])[: args.top]
        for r in sub:
            t = tile(pieces, [concave, tid], [r["placed"]],
                     f"#{tid}  ->  corner {r['i']}-{r['i']+1} of #{concave}",
                     f"ratio {r['Lt1']:.0f}:{r['Lt2']:.0f} vs {r['Ln1']:.0f}:{r['Ln2']:.0f}"
                     f"   resid {r['resid']*100:.1f}%   scale x{r['scale']:.3f}")
            tiles.append(t)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].shape[1], tiles[0].shape[0]
    big = np.full((64 + rows * th + 60, cols * tw + 20, 3), 255, np.uint8)
    cv2.putText(big, f"proportional (ratio) matching - {len(results)} corner fits, no mirror",
                (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        big[64 + r * th:64 + r * th + t.shape[0], 10 + c * tw:10 + c * tw + t.shape[1]] = t
    out_png = outdir / "50_ratio_match.png"
    cv2.imwrite(str(out_png), big)
    print(f"写出 {out_png}  {big.shape[1]}x{big.shape[0]}")

    for r in sorted(results, key=lambda x: x["resid"])[:12]:
        print(f"  #{r['tid']} 角 {r['i']}-{r['i']+1}: 比例 {r['Lt1']:.0f}:{r['Lt2']:.0f} vs "
              f"{r['Ln1']:.0f}:{r['Ln2']:.0f}  残差 {r['resid']*100:.1f}%  缩放 ×{r['scale']:.3f}  "
              f"穿透 {r['ov']*100:.1f}%")
    (outdir / "50_ratio_match.json").write_text(
        json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in r.items()} for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
