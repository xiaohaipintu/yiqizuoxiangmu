"""不规则切割 + 用 拼图还原项目 的 beam_search 拼回去。

和 纹理拼合/ 里的九宫格不同：这里把一张方图用 **Voronoi 分割** 切成 N 块多边形碎片
（每块的边数、形状、面积都不一样），完全不是矩形。然后把它们写成 拼图还原项目
认的 pieces.json 格式，直接调用项目自己的 `src/beam_search.py` 去拼 ——
用的是项目里的那套原理：边长/共线匹配、不许镜像、shapely 精确拒重叠、束搜索。

用法：
    # 单张，看效果（会写出 cut.png / assembled.png / report.json）
    python 纹理拼合/tools/irregular_cut.py --image 纹理拼合/照片/001.jpg --n 6 --out out_irr

    # 批量跑基准
    python 纹理拼合/tools/irregular_cut.py --bench --limit 20 --n 6
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as SPoly, box as sbox
from shapely.ops import unary_union

PROJ = Path(__file__).resolve().parents[2] / "拼图还原项目"
SRC = PROJ / "src"
sys.path.insert(0, str(SRC))
import geometry as G  # noqa: E402
from assemble import Piece, place_edge, world  # noqa: E402


# --------------------------------------------------------------------------
# 1) 不规则切割：Voronoi 分割一个正方形
# --------------------------------------------------------------------------
def voronoi_cut(size: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """把 [0,size]² 切成 n 块不规则多边形（Voronoi cell ∩ 正方形）。

    边界附近的 cell 在 scipy 里可能是无界的，所以在正方形外圈补一圈镜像点，
    让所有关心区域的 cell 都变成有限多边形。
    """
    pts = rng.uniform(0.10, 0.90, size=(n, 2)) * size
    far = size * 2.0
    extra = np.array([[-far, -far], [far, -far], [-far, far], [far, far],
                      [size / 2, -far], [size / 2, far],
                      [-far, size / 2], [far, size / 2]], float)
    vor = Voronoi(np.vstack([pts, extra]))
    square = sbox(0, 0, size, size)

    cells: list[np.ndarray] = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            continue
        poly = SPoly(vor.vertices[region]).buffer(0)
        cell = poly.intersection(square)
        if cell.is_empty:
            continue
        if cell.geom_type == "MultiPolygon":
            cell = max(cell.geoms, key=lambda g: g.area)
        if cell.geom_type != "Polygon" or len(cell.exterior.coords) < 4:
            continue
        cells.append(np.asarray(cell.exterior.coords[:-1], float))
    return cells


def to_pieces_json(cells: list[np.ndarray]) -> dict:
    """转成 拼图还原项目 的 pieces.json 结构（id / polygon / area / centroid / edges）。"""
    pieces = []
    for i, poly in enumerate(cells, start=1):
        if G.polygon_area(poly) < 0:                 # 统一成逆时针
            poly = poly[::-1]
        pieces.append({
            "id": i,
            "polygon": poly.tolist(),
            "area": float(G.polygon_area(poly)),
            "centroid": G.centroid(poly).tolist(),
            "edges": G.edges(poly),
        })
    return {"image": "", "pieces": pieces}


# --------------------------------------------------------------------------
# 2) 拼装：复用项目的 Piece / place_edge / world / poly_overlap_area，
#    但把「状态排序」从凸包空隙换成「对接边长不一致度」。
#
#    为什么必须换：beam_search.py 用 gap_pct（凸包空隙）给**部分拼装**打分。
#    不规则碎片零散摆几块时，凸包天然远大于已摆面积，空隙很大 —— 真解在中途
#    就被剪掉了（实测 6 块时真解根本不在 beam 里）。
#    换成边长一致度后，正确的对接会立刻拿到接近 0 的代价。
# --------------------------------------------------------------------------
def _expand(pieces, placed, polys, used, qid, cov, overlap_tol,
            img=None, tw: float = 0.0, tex_k: float = 0.35):
    """与 beam_search.expand 同构，额外返回这一对的代价。

    代价 = 形状项（边长不一致 + 贴合不完整）+ tw × 纹理项。
    形状项同时充当**候选生成与硬过滤**：边长差太大、贴合不足、穿透超限的直接丢弃，
    纹理无权把一个物理上放不进去的摆法救回来。
    """
    q = pieces[qid]
    out = []
    for k, (pid, ang_a, tx_a, ty_a) in enumerate(placed):
        pw = polys[k]
        p = pieces[pid]
        for ei in range(p.n):
            if ei in used[pid]:
                continue
            aw, bw = pw[ei], pw[(ei + 1) % p.n]
            La = float(np.linalg.norm(bw - aw))
            if La < 1e-6:
                continue
            u = (bw - aw) / La
            for fi in range(q.n):
                eb = q.edges[fi]
                Lb = float(eb["length"])
                for anchor in (True, False):
                    ang, t = place_edge(q, fi, aw, bw, anchor)
                    wq = world(q, ang, t)
                    qaw = (G.rotation(ang) @ np.asarray(eb["a"], float)) + np.asarray(t, float)
                    qbw = (G.rotation(ang) @ np.asarray(eb["b"], float)) + np.asarray(t, float)
                    proj = np.clip([((qaw - aw) @ u), ((qbw - aw) @ u)], 0, La)
                    shared = float(max(0.0, proj.max() - proj.min()))
                    if shared < cov * min(La, Lb):
                        continue
                    ca, cq = p.poly.mean(axis=0), wq.mean(axis=0)
                    sa = u[0] * (ca[1] - aw[1]) - u[1] * (ca[0] - aw[0])
                    sq = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                    if sa * sq >= 0:
                        continue
                    ov = sum(G.poly_overlap_area(wq, pp) for pp in polys)
                    if ov > overlap_tol * q.area:
                        continue
                    # --- 形状项 ---
                    seam = abs(La - Lb) / max(La, Lb)
                    seam += 0.5 * (1.0 - min(1.0, shared / min(La, Lb)))
                    # --- 纹理项（可选）：形状已放行之后，才用它区分"长度像但拼错" ---
                    if img is not None and tw:
                        seam += tw * _texture_cost(
                            img, (ang_a, tx_a, ty_a), (ang, float(t[0]), float(t[1])),
                            aw, bw, 1.0 if sa > 0 else -1.0, k=tex_k)
                    out.append((pid, ei, fi, ang, float(t[0]), float(t[1]), wq, seam))
    return out


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """双线性采样（pts 为原图像素坐标，Nx2）。"""
    h, w = img.shape[:2]
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return top * (1 - wy) + bot * wy


def _texture_cost(img, pose_a, pose_b, aw, bw, side, k: float = 0.35,
                  samples: int = 24) -> float:
    """看接缝两侧的像素是否连续（跨缝差 ÷ 块内差）。

    踩过两次坑，都记在这里：

    1) 一开始直接比"跨缝两点的色差"，真邻接也判成 0.54~0.96。因为 Voronoi 切线
       本身常常压在照片的硬边上，跨 3 像素颜色就能差 2.5 倍 —— 这和拼得对不对无关。
       所以改成用**块内自身变化**做基线：跨缝跳变 ≈ 块内跳变 → 比值 ≈ 1。
    2) 采样距离 k 也不能大。k=1.6 时两点相距 3.2px，测的还是"图像在这 3px 里变化多大"，
       实测分离度只有 0.30×（完全反了）；k=0.35（两点相距 0.7px，等于相邻像素）
       分离度 21×：真邻接比值最大 4.2，假配对最小 88.5。

    返回 ratio/(ratio+20) 压到 0~1：真邻接 ≈0.13，假配对 ≈0.8~0.95。
    """
    u = (bw - aw) / (np.linalg.norm(bw - aw) + 1e-12)
    n = np.array([-u[1], u[0]], float) * side        # 指向已放那块的内侧
    ts = np.linspace(0.12, 0.88, samples)[:, None]
    base = aw[None, :] + (bw - aw)[None, :] * ts     # (N,2) 接缝上的点

    def to_orig(world_pts, pose):
        ang, tx, ty = pose
        Rm = G.rotation(-ang)
        return (Rm @ (world_pts - np.array([tx, ty], float)).T).T

    def sq(a, b):
        d = a - b
        return float(np.mean(np.sum(d * d, axis=1)))

    pa1 = _sample(img, to_orig(base + n * k, pose_a))
    pa2 = _sample(img, to_orig(base + n * 2 * k, pose_a))
    pb1 = _sample(img, to_orig(base - n * k, pose_b))
    pb2 = _sample(img, to_orig(base - n * 2 * k, pose_b))

    cross = sq(pa1, pb1)
    baseline = 0.5 * (sq(pa1, pa2) + sq(pb1, pb2)) + 1.0
    ratio = cross / baseline
    return ratio / (ratio + 20.0)



def _rect_fill(polys: list[np.ndarray]) -> float:
    """拼装结果和它的最小外接矩形有多贴合（0~1）。4 条撕裂纸条拼对了就该接近 1。"""
    ps = [SPoly(p).buffer(0) for p in polys]
    u = unary_union(ps)
    rect = u.minimum_rotated_rectangle
    return float(u.area / rect.area) if getattr(rect, "area", 0) else 0.0


def _union_stats(polys: list[np.ndarray]) -> dict:
    """并集的三个量：覆盖率（最大连通块占全部面积）、矩形贴合、空洞。

    只看「矩形贴合」会被钻空子 —— 四块并排摆着但各不相连，并集的最小外接矩形
    照样很大，贴合率能到 96%。所以必须同时要求**接成一片**（覆盖率≈1）。
    """
    ps = [SPoly(p).buffer(0) for p in polys]
    total = sum(p.area for p in ps) or 1.0
    u = unary_union(ps)
    if u.geom_type == "MultiPolygon":
        parts = sorted(u.geoms, key=lambda g: -g.area)
        main, n_parts = parts[0], len(parts)
    else:
        main, n_parts = u, 1
    rect = main.minimum_rotated_rectangle
    holes = sum(abs(SPoly(r).area) for r in getattr(main, "interiors", []))
    # 内部接触边总长：拼对了，块与块之间的共线接触最长（这是拼图最本质的判据）
    shared = 0.0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            inter = ps[i].intersection(ps[j])
            if inter.geom_type == "LineString":
                shared += inter.length
            elif inter.geom_type == "MultiLineString":
                shared += sum(g.length for g in inter.geoms)
    return {
        "coverage": float(main.area / total),
        "rect_fill": float(main.area / rect.area) if getattr(rect, "area", 0) else 0.0,
        "holes_pct": float(holes / total * 100.0),
        "components": n_parts,
        "shared_px": float(shared),
        "shared_ratio": float(shared / (sum(p.length for p in ps) or 1.0)),
    }



def assemble(pieces: dict, beam: int = 80, time_budget: float = 45.0,
             cov: float = 0.6, overlap_tol: float = 0.004,
             img=None, tw: float = 0.0, tex_k: float = 0.35,
             rank: str = "joined", collect: int = 0) -> dict | None:
    """束搜索拼装。

    rank 决定**在完整解之间**怎么选：
      "joined" 内部接触边总长最大（默认，拼图最本质的判据）
      "seam"   累计接缝代价最小（局部最优 —— 会被"局部拼得很好但整体错"骗过）
      "rect"   最小外接矩形贴合率最高
      "mixed"  seam 与 rect 的加权

    为什么默认是 "joined"：真实的撕纸拼图里，#1+#3 和 #2+#4 可能各自拼得很好
    （局部边长代价很低），但要把两个"整体"再接起来，需要整体相对旋转 ——
    局部的边长代价给不出这个偏好，会停在「差一条边没接上」的解。
    实测（手撕拼图.jpg，4 块）：
      rank=seam   → 接触边 950px，2 条缝，空隙 2.89%，#2 旋转 155°（错）
      rank=rect   → 接触边 36px（只靠角点相碰），空隙 1.69%（更差）
      rank=joined → 接触边 1400px，3 条完整缝（485/465/450px），空隙 0.79%，#2 旋转 336°（对）
    """
    ids = list(pieces)
    n = len(ids)
    seed = max(ids, key=lambda i: pieces[i].area)
    states = [([(seed, 0.0, 0.0, 0.0)],
               [world(pieces[seed], 0.0, (0.0, 0.0))],
               {pid: set() for pid in ids}, 0.0)]
    deadline = time.time() + time_budget
    complete: list[tuple] = []          # 所有见到的完整解（去重后）

    for level in range(2, n + 1):
        cands = []
        for placed, polys, used, cost in states:
            done = {p[0] for p in placed}
            for qid in ids:
                if qid in done:
                    continue
                for pid, ei, fi, ang, tx, ty, wq, seam in _expand(
                        pieces, placed, polys, used, qid, cov, overlap_tol,
                        img=img, tw=tw, tex_k=tex_k):
                    nu = {k: set(v) for k, v in used.items()}
                    nu[pid].add(ei)
                    nu[qid] = {fi}
                    cands.append((placed + [(qid, ang, tx, ty)],
                                  polys + [wq], nu, cost + seam))
            if time.time() > deadline:
                break
        if not cands:
            break
        cands.sort(key=lambda s: s[3])
        # 去重：同一套摆放（坐标四舍五入后一致）只留代价最低的
        seen, keep = set(), []
        for s in cands:
            sig = tuple(sorted((p[0], round(p[1], 1), round(p[2], 1), round(p[3], 1))
                               for p in s[0]))
            if sig in seen:
                continue
            seen.add(sig)
            keep.append(s)
            if len(keep) >= beam:
                break
        states = keep
        if level == n:
            complete = keep[:max(collect, beam)]
        if time.time() > deadline:
            break

    full = [s for s in complete if len(s[0]) == n] or \
           [s for s in states if len(s[0]) == n]
    if not full:
        return None

    # ---- 在完整解之间排序 ----
    seams = [s[3] for s in full]
    lo, hi = (min(seams), max(seams)) if seams else (0.0, 1.0)

    def norm_seam(v: float) -> float:
        return (v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.0

    def score(s) -> float:
        if rank == "seam":
            return norm_seam(s[3])
        st = _union_stats(s[1])
        if rank == "rect":
            # 必须接成一片（coverage≈1）且贴合矩形；否则四块悬空摆也能拿高分
            return (1.0 - st["coverage"]) * 3.0 + (1.0 - st["rect_fill"]) + st["holes_pct"] / 100.0
        if rank == "joined":
            # 主判据：内部接触边总长越大越好（拼图最本质），缺口/不连通重罚
            pen = (1.0 - st["coverage"]) * 3.0 + st["holes_pct"] / 100.0
            return -st["shared_ratio"] + pen
        return (0.4 * norm_seam(s[3]) + (1.0 - st["coverage"]) * 3.0
                + (1.0 - st["rect_fill"]))                     # mixed

    full.sort(key=score)
    placed, polys, _used, cost = full[0]
    st = _union_stats(polys)
    return {"placed": [list(p) for p in placed],
            "polys": [p.tolist() for p in polys],
            "seam": cost,
            "gap_pct": round(_gap_pct(polys), 2),
            "rect_fill": round(st["rect_fill"], 4),
            "coverage": round(st["coverage"], 4),
            "components": st["components"],
            "shared_px": round(st["shared_px"], 1),
            "shared_ratio": round(st["shared_ratio"], 4),
            "n_candidates": len(full)}



def _gap_pct(polys: list[np.ndarray]) -> float:
    pts = np.vstack(polys).astype(np.float32)
    hull = abs(float(cv2.contourArea(cv2.convexHull(pts))))
    s = sum(abs(float(G.polygon_area(p))) for p in polys)
    return max(0.0, hull - s) / max(s, 1e-9) * 100.0
def run_beam(pieces_json: Path, outdir: Path, time_budget: float,
             beam: int = 200, cov: float = 0.35,
             overlap: float = 0.006) -> dict | None:
    cmd = [sys.executable, str(SRC / "beam_search.py"),
           "--pieces", str(Path(pieces_json).resolve()), "--out", str(Path(outdir).resolve()),
           "--beam", str(beam), "--time", str(float(time_budget)),
           "--cov", str(cov), "--overlap", str(overlap)]
    proc = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=time_budget + 180)
    if proc.returncode != 0:
        return {"error": (proc.stdout or "")[-500:] + (proc.stderr or "")[-500:]}
    sol = outdir / "beam_solutions.json"
    if not sol.exists():
        return {"error": "没有产出 beam_solutions.json"}
    doc = json.loads(sol.read_text(encoding="utf-8"))
    sols = doc.get("solutions") or []
    return {"solutions": sols, "n": doc.get("n", len(sols)), "log": proc.stdout[-800:]}


# --------------------------------------------------------------------------
# 3) 判定：刚体不变 —— 两两质心距离应与原图一致
# --------------------------------------------------------------------------
def centroid_dist_matrix(polys: list[np.ndarray]) -> np.ndarray:
    c = np.array([np.asarray(p, float).mean(axis=0) for p in polys])
    d = c[:, None, :] - c[None, :, :]
    return np.linalg.norm(d, axis=2)


def verify(cells: list[np.ndarray], sol: dict) -> dict:
    """按 placed 里的 id 顺序取世界多边形，比两两质心距离矩阵。"""
    polys = [np.asarray(p, float) for p in sol["polys"]]
    ids = [int(it[0]) for it in sol["placed"]]
    truth = [cells[i - 1] for i in ids]
    d_pred = centroid_dist_matrix(polys)
    d_true = centroid_dist_matrix(truth)
    scale = float(np.max(d_true)) or 1.0
    err = float(np.max(np.abs(d_pred - d_true))) / scale
    return {"max_rel_dist_err": err, "ok": err < 0.02}


# --------------------------------------------------------------------------
# 4) 可视化
# --------------------------------------------------------------------------
def _imwrite(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite 在 Windows 下写不了含中文的路径，统一用 imencode + tofile。"""
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if ok:
        buf.tofile(str(path))


def draw_cut(cells: list[np.ndarray], img_path: Path | None, size: int, path: Path) -> None:
    base = np.full((size, size, 3), 24, np.uint8)
    src = None
    if img_path and img_path.exists():
        src = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if src is not None:
            src = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
            base = (src * 0.35).astype(np.uint8)
    palette = [(235, 130, 60), (60, 160, 240), (90, 200, 120), (230, 160, 50),
               (200, 90, 220), (60, 220, 230), (180, 70, 180), (250, 150, 90),
               (120, 200, 90), (60, 120, 255)]
    for i, poly in enumerate(cells):
        p = np.round(poly).astype(np.int32)
        cv2.polylines(base, [p], True, palette[i % len(palette)], 2, cv2.LINE_AA)
        c = poly.mean(axis=0).astype(int)
        cv2.putText(base, str(i + 1), (c[0] - 8, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
    _imwrite(path, base)


def draw_assembled(polys: list[np.ndarray], path: Path) -> None:
    pts = np.vstack([np.asarray(p, float) for p in polys])
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    pad = 20.0
    w = int(mx[0] - mn[0] + 2 * pad)
    h = int(mx[1] - mn[1] + 2 * pad)
    canvas = np.full((h, w, 3), 24, np.uint8)
    palette = [(235, 130, 60), (60, 160, 240), (90, 200, 120), (230, 160, 50),
               (200, 90, 220), (60, 220, 230), (180, 70, 180), (250, 150, 90),
               (120, 200, 90), (60, 120, 255)]
    for i, poly in enumerate(polys):
        p = np.asarray(poly, float) - mn + pad
        cv2.fillPoly(canvas, [np.round(p).astype(np.int32)],
                     palette[i % len(palette)])
        cv2.polylines(canvas, [np.round(p).astype(np.int32)], True, (245, 245, 245),
                      2, cv2.LINE_AA)
    _imwrite(path, canvas)


# --------------------------------------------------------------------------
# 5) 单张 / 批量
# --------------------------------------------------------------------------
def _load_square(path: Path | None, size: int) -> np.ndarray:
    """读成 size×size 的 RGB（碎片坐标就定义在这个尺度上）。"""
    if path is None:
        return np.full((size, size, 3), 128, np.uint8)
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return np.full((size, size, 3), 128, np.uint8)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)


def one(path: Path | None, n: int, size: int, time_budget: float, outdir: Path,
        seed: int, use_beam: bool = False,
        texture: bool = False, tw: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    cells = voronoi_cut(size, n, rng)
    if len(cells) < 2:
        return {"ok": False, "error": "切割失败"}

    outdir.mkdir(parents=True, exist_ok=True)
    data = to_pieces_json(cells)
    pj = outdir / "pieces.json"
    pj.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    draw_cut(cells, path, size, outdir / "cut.png")

    t0 = time.time()
    if use_beam:                       # 对照：项目原来的 beam_search（按凸包空隙排序）
        res = run_beam(pj, outdir, time_budget)
        if not res or "error" in res:
            return {"ok": False, "error": (res or {}).get("error", "beam 失败")}
        sols = res.get("solutions") or []
        if not sols:
            return {"ok": False, "error": "没有完整拼法"}
        sol = sols[0]
        n_sol = res.get("n", len(sols))
    else:                              # 本项目：按对接边长一致度排序
        pieces = {int(d["id"]): Piece(d) for d in data["pieces"]}
        img = _load_square(path, size) if texture else None
        sol = assemble(pieces, beam=80, time_budget=max(10.0, time_budget * 3),
                       img=img, tw=tw)
        if sol is None:
            return {"ok": False, "error": "没有完整拼法"}
        n_sol = 1

    cost = time.time() - t0
    verd = verify(cells, sol)
    draw_assembled(sol["polys"], outdir / "assembled.png")
    (outdir / "report.json").write_text(
        json.dumps({"ok": verd["ok"], "max_rel_dist_err": verd["max_rel_dist_err"],
                    "seam": sol.get("seam"), "gap_pct": sol.get("gap_pct"),
                    "pieces": len(cells), "seconds": round(cost, 1)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": verd["ok"], "max_rel_dist_err": verd["max_rel_dist_err"],
            "gap_pct": sol.get("gap_pct"), "seam": sol.get("seam"),
            "n_sol": n_sol, "seconds": round(cost, 1), "pieces": len(cells)}


def main() -> int:
    ap = argparse.ArgumentParser(description="不规则切割 + 拼图还原项目 beam_search 拼回")
    ap.add_argument("--image", default=None, help="输入方图；不给就切空白色块")
    ap.add_argument("--n", type=int, default=6, help="切几块（不规则多边形）")
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--out", default=None, help="输出目录（默认临时目录）")
    ap.add_argument("--time", type=float, default=6.0, help="每张图的束搜索预算（秒）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bench", action="store_true", help="跑 照片/ 目录的批量基准")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--use-beam", action="store_true",
                    help="对照组：改用项目原来的 beam_search.py（按凸包空隙排序，预期会拼错）")
    ap.add_argument("--texture", action="store_true",
                    help="形状过滤之后再叠加纹理代价（默认关，纯形状）")
    ap.add_argument("--texture-weight", type=float, default=1.0,
                    help="纹理项权重 tw（代价 = 形状项 + tw × 纹理项）")
    args = ap.parse_args()

    if args.bench:
        photos = sorted(Path("纹理拼合/照片").glob("*.jpg"))[: args.limit]
        ok = 0
        print("%-10s %-6s %-8s %-10s %s" % ("照片", "块数", "耗时", "质心误差", "结果"))
        print("-" * 56)
        for i, f in enumerate(photos):
            out = Path(tempfile.gettempdir()) / f"irr_{f.stem}"
            r = one(f, args.n, args.size, args.time, out, seed=i + 1,
                    use_beam=args.use_beam, texture=args.texture,
                    tw=args.texture_weight)
            if r.get("ok"):
                ok += 1
                print("%-10s %-6d %-8s %-10s ✓ %d 种拼法，空隙 %s%%"
                      % (f.name, r["pieces"], str(r["seconds"]) + "s",
                         "%.4f" % r["max_rel_dist_err"], r["n_sol"], r["gap_pct"]))
            else:
                print("%-10s %-6s %-8s %-10s ✗ %s"
                      % (f.name, r.get("pieces", "-"), r.get("seconds", "-"),
                         "%.4f" % r.get("max_rel_dist_err", 0) if "max_rel_dist_err" in r else "-",
                         r.get("error", "")))
        print()
        print("成功率 %d/%d = %.0f%%" % (ok, len(photos), 100.0 * ok / max(1, len(photos))))
        return 0

    outdir = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "irregular_demo"
    r = one(Path(args.image) if args.image else None, args.n, args.size,
            args.time, outdir, args.seed, use_beam=args.use_beam,
            texture=args.texture, tw=args.texture_weight)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print("输出目录：", outdir)
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
