"""把碎片拼成一个整体（仅按边长匹配，默认不镜像）。

匹配规则（边长匹配 + 拆纸的物理事实）：
    * 两块贴合时，两条边必须共线、方向相反、贴着同一侧（不镜像时不翻面）；
    * 贴合段至少覆盖较短那条边的一半，避免"擦边"的假贴合；
    * 允许一条边的端点落在另一条边的内部 —— 这就是拆解时的 T 型接缝，
      一块的顶点正好压在另一块的边上，于是两条边等于一条长边（如 1334+621≈1962）。

排序（只用于选"哪一种是真正的拼法"，不参与匹配判定）：
    空穴率↑ → 外轮廓周长↑ → 完全等长的接缝数↓

用法:
    python src/assemble.py --pieces out2/pieces.json --image input/pieces2.jpg --out out2
输出:
    out2/solution.json    最优拼法
    out2/10_assembled.png 拼好的图
    out2/11_candidates.png 前若干候选并排（便于核对）
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

PALETTE = [
    (56, 56, 224), (0, 140, 255), (60, 180, 75), (200, 120, 20),
    (200, 60, 200), (30, 200, 200), (140, 60, 160), (0, 90, 200),
]


class Piece:
    def __init__(self, d):
        self.id = int(d["id"])
        self.poly = np.array(d["polygon"], float)
        self.area = float(d["area"])
        self.edges = list(d["edges"])
        self.n = len(self.poly)
        self.centroid = np.array(d["centroid"], float)


def world(piece: Piece, deg: float, t, mirror=False, scale=1.0) -> np.ndarray:
    S = np.array([[-1.0, 0.0], [0.0, 1.0]]) if mirror else np.eye(2)
    M = (G.rotation(deg) @ S) * scale
    return (M @ piece.poly.T).T + np.asarray(t, float)


def place_edge(piece: Piece, e_idx: int, ta, tb, anchor_a: bool):
    """把 piece 的第 e_idx 条边的 a 端贴到 ta（或 a 端贴 tb），方向反向，返回 (角度, 平移)。"""
    e = piece.edges[e_idx]
    a = np.array(e["a"], float)
    b = np.array(e["b"], float)
    ds = b - a
    ds /= np.linalg.norm(ds) + 1e-12
    dt = np.asarray(tb, float) - np.asarray(ta, float)
    dt /= np.linalg.norm(dt) + 1e-12
    # 反向贴合
    want = -dt
    dst = np.asarray(tb, float) if anchor_a else np.asarray(ta, float)
    ang = math.degrees(math.atan2(want[1], want[0]) - math.atan2(ds[1], ds[0])) % 360.0
    return ang, dst - G.rotation(ang) @ a


def overlap_len(seg1, seg2) -> float:
    """两条共线线段的公共长度（按第一条的投影参数）。"""
    a1, b1 = np.asarray(seg1[0], float), np.asarray(seg1[1], float)
    a2, b2 = np.asarray(seg2[0], float), np.asarray(seg2[1], float)
    u = (b1 - a1) / (np.linalg.norm(b1 - a1) + 1e-12)
    t2 = [(p - a1) @ u for p in (a2, b2)]
    lo, hi = min(t2), max(t2)
    L1 = np.linalg.norm(b1 - a1)
    return float(max(0.0, min(hi, L1) - max(lo, 0.0)))


class Enum:
    def __init__(self, pieces, min_cov=0.5, overlap_tol=0.006, mirror=False):
        self.pieces = {p.id: p for p in pieces}
        self.min_cov = min_cov
        self.overlap_tol = overlap_tol
        self.mirror = mirror

    def candidates(self, placed, polys, qid):
        """把 qid 贴到已放簇的任意一条边上，返回所有可行状态。"""
        q = self.pieces[qid]
        used = {r[0] for r in placed}
        out = []
        for k, (pid, deg, tx, ty) in enumerate(placed):
            pw = polys[k]
            p = self.pieces[pid]
            cw = pw.mean(axis=0)
            for ei in range(p.n):
                aw, bw = pw[ei], pw[(ei + 1) % p.n]
                La = float(np.linalg.norm(bw - aw))
                if La < 1e-6:
                    continue
                u = (bw - aw) / La
                for fi in range(q.n):
                    e = q.edges[fi]
                    Lb = float(e["length"])
                    for anchor in (True, False):
                        ang, t = place_edge(q, fi, aw, bw, anchor)
                        wq = world(q, ang, t)
                        # 贴合段长度：新块的这条边与目标边的公共长度
                        qa = np.array([e["a"]], float)
                        qb = np.array([e["b"]], float)
                        qaw = (G.rotation(ang) @ qa.T).T[0] + t
                        qbw = (G.rotation(ang) @ qb.T).T[0] + t
                        shared = overlap_len((aw, bw), (qaw, qbw))
                        if shared < self.min_cov * min(La, Lb):
                            continue
                        # 必须在目标边另一侧
                        cq = wq.mean(axis=0)
                        side_p = u[0] * (cw[1] - aw[1]) - u[1] * (cw[0] - aw[0])
                        side_q = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                        if side_p * side_q >= 0:
                            continue
                        ov = 0.0
                        for sp in polys:
                            ov += G.convex_overlap_area(wq, sp)
                        if ov > self.overlap_tol * q.area:
                            continue
                        new_placed = placed + [(qid, ang, float(t[0]), float(t[1]))]
                        out.append((new_placed, polys + [wq], shared))
        return out

    def run(self):
        seed = max(self.pieces, key=lambda i: self.pieces[i].area)
        init = world(self.pieces[seed], 0.0, (0.0, 0.0))
        stack = [([(seed, 0.0, 0.0, 0.0)], [init], [])]
        results = []
        while stack:
            placed, polys, joins = stack.pop()
            if len(placed) == len(self.pieces):
                results.append((placed, polys, joins))
                continue
            used = {r[0] for r in placed}
            for qid in self.pieces:
                if qid in used:
                    continue
                for np_, npolys, shared in [(x[0], x[1], x[2]) for x in self.candidates(placed, polys, qid)]:
                    stack.append((np_, npolys, joins + [shared]))
        return results


def score(placed, polys, joins) -> dict:
    pts = np.vstack(polys)
    hull_pts = cv2.convexHull(pts.astype(np.float32))
    hull_area = abs(float(cv2.contourArea(hull_pts)))
    hull_peri = float(cv2.arcLength(hull_pts, True))
    sum_area = sum(G.polygon_area(p) for p in polys)
    holes = max(0.0, hull_area - sum_area)
    nvert = len(G.drop_collinear(hull_pts.reshape(-1, 2).astype(float), tol_deg=3.0))
    return {
        "holes_pct": holes / sum_area * 100.0,
        "hull_area": hull_area,
        "hull_perimeter": hull_peri,
        "sum_area": sum_area,
        "hull_vertices": nvert,
        "joins": joins,
    }


def draw(assembly_polys, ids, size=1100, margin=40, fill=True):
    allp = np.vstack(assembly_polys)
    w = allp[:, 0].max() - allp[:, 0].min()
    h = allp[:, 1].max() - allp[:, 1].min()
    s = (size - 2 * margin) / max(w, h)
    W = int(w * s) + 2 * margin
    H = int(h * s) + 2 * margin
    canvas = np.full((H, W, 3), 255, np.uint8)
    mn = np.array([allp[:, 0].min(), allp[:, 1].min()])
    for poly, pid in zip(assembly_polys, ids):
        pts = ((poly - mn) * s + margin).astype(np.int32)
        if fill:
            cv2.fillPoly(canvas, [pts], PALETTE[(pid - 1) % len(PALETTE)])
        cv2.polylines(canvas, [pts], True, (35, 35, 35), 2, cv2.LINE_AA)
        c = pts.mean(axis=0).astype(int)
        cv2.putText(canvas, f"#{pid}", (c[0] - 12, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"#{pid}", (c[0] - 12, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out2/pieces.json")
    ap.add_argument("--image", default="input/pieces2.jpg")
    ap.add_argument("--out", default="out2")
    ap.add_argument("--min-cov", type=float, default=0.5)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = [Piece(d) for d in data["pieces"]]
    en = Enum(pieces, min_cov=args.min_cov, mirror=args.mirror)
    results = en.run()
    print(f"枚举到 {len(results)} 个连成一体的拼法")

    scored = []
    for r in results:
        s = score(*r)
        scored.append((s, r))
    scored.sort(key=lambda t: (round(t[0]["holes_pct"], 2), t[0]["hull_perimeter"]))
    print(f"{'名次':>4} {'空穴%':>7} {'外轮廓周长':>10} {'顶点数':>6} {'接缝长':>22}")
    for i, (s, r) in enumerate(scored[:10], 1):
        joins = " ".join(f"{v:.0f}" for v in s["joins"])
        print(f"{i:>4} {s['holes_pct']:>7.2f} {s['hull_perimeter']:>10.0f} {s['hull_vertices']:>6} {joins:>22}")

    best_s, best_r = scored[0]
    placed, polys, joins = best_r

    # 归一化并输出
    pts = np.vstack(polys)
    off = np.array([pts[:, 0].min(), pts[:, 1].min()])
    sol = {"pieces": [{
        "id": pid, "rotate_deg": float(deg % 360.0), "mirror": False,
        "translate": [float(tx - off[0]), float(ty - off[1])],
        "polygon_world": (w - off).tolist(),
    } for (pid, deg, tx, ty), w in zip(placed, polys)]}
    (outdir / "solution.json").write_text(json.dumps(sol, ensure_ascii=False, indent=2), encoding="utf-8")

    img = draw(polys, [r[0] for r in placed], size=1100)
    cv2.imwrite(str(outdir / "10_assembled.png"), img)
    print(f"写出 {outdir / '10_assembled.png'} {img.shape[1]}x{img.shape[0]}")

    # 候选对比图
    if len(scored) > 1:
        tiles = scored[: args.top]
        box = 420
        cols = min(3, len(tiles))
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((80 + rows * (box + 40), 20 + cols * (box + 10), 3), 255, np.uint8)
        cv2.putText(sheet, "candidates ranked by holes / outline length", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        for i, (s, r) in enumerate(tiles):
            rr, cc = divmod(i, cols)
            x = 10 + cc * (box + 10)
            y = 70 + rr * (box + 40)
            im = draw(r[1], [q[0] for q in r[0]], size=box - 16, margin=8)
            h_, w_ = im.shape[:2]
            sheet[y:y + h_, x:x + w_] = im
            cv2.putText(sheet, f"#{i+1} holes {s['holes_pct']:.1f}%", (x, y + box + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.imwrite(str(outdir / "11_candidates.png"), sheet)
        print(f"写出 {outdir / '11_candidates.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
