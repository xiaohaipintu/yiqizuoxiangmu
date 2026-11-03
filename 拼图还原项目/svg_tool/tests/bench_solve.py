"""诚实评测台 —— 只给碎块（带顶点噪声），不给原图；拼完再跟真值比。

为什么要这个台子
--------------
`jigsaw_poly_solve` 报的「60/60 完美复原」是**自己 Voronoi 切、自己拼**：相邻两片的
共享边精确等长（误差 ~1e-13），题目天然带着「精确互补边」这个答案线索，等于把答案揉进题目。
这个台子把线索掐掉：

  ① **顶点加噪**：每片碎片的每个顶点各自加高斯噪声（σ 像素）。共享边从此不再等长，
     长度误差 ~1~3 px —— 相当于真实抠图 / 手撕造成的轮廓误差；
  ② **只给碎块**：求解器拿到的是「散落图 + 抠出来的多边形」，**不传原图、不传真值位姿**；
  ③ **真值打分**：拼完先做全局刚体对齐（以每片为锚各试一次，取拼对最多的那个锚），
     再逐片跟真值多边形算 IoU → 得到真正的「拼对率」。

判定「拼对」用 IoU > 0.75（加噪本身让轮廓差 1~2 px，1.0 不现实）。

用法
----
    python svg_tool/tests/bench_solve.py --n 60 --noise 0 --solver v2
    python svg_tool/tests/bench_solve.py --n 60 --noise 1.5 --solver v1 --v1-len-tol 6
    python svg_tool/tests/bench_solve.py --sweep          # v1 vs v2 在 σ=0/1.5/3 全表
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve()
PROJ = HERE.parents[2]                       # 拼图还原项目/
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "svg_tool" / "tests"))

import jigsaw_poly_solve as J                 # noqa: E402
import jigsaw_solve_v2 as V2                  # noqa: E402


# ------------------------------------------------------------------ 造题
def build_case(n: int, S: int, tex: np.ndarray, seed: int, noise: float,
               side_scale: float = 1.9):
    """切 n 片 → 每片顶点加噪 → 随机旋转平移散开 → 渲染散落图。

    返回 (polys 散落坐标, gt_pose 每片 (Rg,tg) 把原图坐标→散落坐标,
          scat 散落图, n_cells)
    """
    cells = [np.asarray(c, float) for c in J.make_cells(n, S, seed)]
    rng = np.random.default_rng(seed + 12345)
    noisy = []
    for c in cells:
        c2 = c + rng.normal(0.0, noise, c.shape) if noise > 0 else c.copy()
        noisy.append(c2)

    side = int(S * side_scale)
    polys, poses = J.scatter(noisy, side, seed)     # poses[i] = (theta, center, cen)

    # 散落图：内容取原纹理，形状按加噪后的多边形
    scat = np.zeros((side, side, 3), np.uint8)
    gt_pose = []
    for i, (theta, center, cen) in enumerate(poses):
        R = J.rot(theta)
        M = np.array([[R[0, 0], R[0, 1], center[0] - (R @ cen)[0]],
                      [R[1, 0], R[1, 1], center[1] - (R @ cen)[1]]], float)
        warped = cv2.warpAffine(tex, M, (side, side), flags=cv2.INTER_LINEAR)
        m = np.zeros((side, side), np.uint8)
        cv2.fillPoly(m, [np.round(polys[i]).astype(np.int32)], 255)
        scat[m > 0] = warped[m > 0]
        gt_pose.append((R, center - R @ cen))       # x_scat = Rg x_orig + tg
    return polys, noisy, gt_pose, scat, side


# ------------------------------------------------------------------ 打分
def score(polys, cells, gt_pose, pose_pred, S: int, thresh: float = 0.75) -> dict:
    """全局刚体对齐 + 逐片 IoU → 真·拼对率。"""
    K = len(cells)
    placed = [k for k in range(K) if pose_pred[k] is not None]
    if not placed:
        return {"placed": 0, "correct": 0, "rate": 0.0, "mean_iou": 0.0}

    gt_lab = np.zeros((S, S), np.int32)
    for k, c in enumerate(cells):
        cv2.fillPoly(gt_lab, [np.round(np.asarray(c)).astype(np.int32)], k + 1)
    ngt = np.bincount(gt_lab.ravel(), minlength=K + 1).astype(np.float64)
    keys = gt_lab.ravel().astype(np.int64) * (K + 1)

    best = {"placed": len(placed), "correct": 0, "rate": 0.0, "mean_iou": 0.0}
    pred_lab = np.zeros((S, S), np.int32)
    for a in placed:                                # 每片都当一次锚，取拼对最多的
        Ra, ta = pose_pred[a]
        Rg, tg = gt_pose[a]
        RG = Rg.T @ Ra.T
        tG = -(Rg @ (Ra.T @ ta + tg))
        pred_lab[:] = 0
        for k in placed:
            # 预测多边形（散落坐标）→ 对齐到原图坐标
            Rk, tk = pose_pred[k]
            q = (np.asarray(polys[k], float) @ Rk.T + tk) @ RG.T + tG
            if not np.isfinite(q).all() or np.abs(q).max() > 1e6:
                continue
            xs, ys = q[:, 0], q[:, 1]
            if xs.max() < 0 or ys.max() < 0 or xs.min() > S or ys.min() > S:
                continue
            cv2.fillPoly(pred_lab, [np.round(q).astype(np.int32)], k + 1)
        bc = np.bincount(keys + pred_lab.ravel().astype(np.int64),
                         minlength=(K + 1) * (K + 1)).reshape(K + 1, K + 1)
        inter = np.diag(bc)[1:]
        npred = bc.sum(axis=0)[1:]
        iou = inter / np.maximum(ngt[1:] + npred - inter, 1.0)
        cnt = int((iou > thresh).sum())
        sc = float(iou.mean())
        if (cnt, sc) > (best["correct"], best["mean_iou"]):
            best.update({"correct": cnt, "rate": cnt / K, "mean_iou": sc})
    return best


# ------------------------------------------------------------------ 渲染对比图
def montage(scat, assembled, gt, S):
    def fit(im, s=420):
        h, w = im.shape[:2]
        f = s / max(h, w, 1)
        im = cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))))
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return im.astype(np.uint8)

    parts = [fit(scat), fit(assembled), fit(gt)]
    hh = max(p.shape[0] for p in parts)
    parts = [cv2.copyMakeBorder(p, 0, hh - p.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0)) for p in parts]
    return cv2.hconcat(parts)


def assemble_image(scat, polys, pose_pred, side, pad=40):
    placed = [k for k in range(len(polys)) if pose_pred[k] is not None]
    if not placed:
        return np.zeros((side, side, 3), np.uint8)
    allp = np.vstack([np.asarray(polys[k], float) @ pose_pred[k][0].T + pose_pred[k][1]
                      for k in placed])
    x0 = int(np.floor(allp[:, 0].min())) - pad
    y0 = int(np.floor(allp[:, 1].min())) - pad
    x1 = int(np.ceil(allp[:, 0].max())) + pad
    y1 = int(np.ceil(allp[:, 1].max())) + pad
    W, H = x1 - x0, y1 - y0
    canvas = np.zeros((H, W, 3), np.uint8)
    for k in placed:
        R, t = pose_pred[k]
        p = np.asarray(polys[k], float) @ R.T + t
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(p - [x0, y0]).astype(np.int32)], 255)
        sx0 = max(0, int(np.floor(np.asarray(polys[k])[:, 0].min())) - 2)
        sy0 = max(0, int(np.floor(np.asarray(polys[k])[:, 1].min())) - 2)
        sx1 = min(side, int(np.ceil(np.asarray(polys[k])[:, 0].max())) + 2)
        sy1 = min(side, int(np.ceil(np.asarray(polys[k])[:, 1].max())) + 2)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        crop = scat[sy0:sy1, sx0:sx1]
        shift = R @ np.array([sx0, sy0], float)
        M = np.array([[R[0, 0], R[0, 1], t[0] + shift[0] - x0],
                      [R[1, 0], R[1, 1], t[1] + shift[1] - y0]], float)
        warped = cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_LINEAR)
        canvas[m > 0] = warped[m > 0]
    return canvas


# ------------------------------------------------------------------ 单次评测
def run_one(solver: str, n: int, noise: float, seed: int, tex, S: int,
            v1_len_tol: float, v1_tex_tol: float, save: pathlib.Path | None):
    t0 = time.time()
    polys, cells, gt_pose, scat, side = build_case(n, S, tex, seed, noise)
    t_case = time.time() - t0
    t0 = time.time()
    if solver == "v1":
        pose, placed, pp = J.solve(polys, len_tol=v1_len_tol, scat=scat,
                                   shape_tol=12.0, tex_tol=v1_tex_tol)
        pose = list(pose)
    else:
        pose, placed, pp = V2.solve(polys, scat, verbose=False)
        pose = list(pose)
    t_solve = time.time() - t0

    met = score(polys, cells, gt_pose, pose, side)
    hull = J.evaluate(pp, side).get("hull_frac")
    row = {"solver": solver, "n": len(polys), "noise": noise,
           "placed": placed, "correct": met["correct"],
           "rate": round(met["rate"] * 100, 1),
           "mean_iou": round(met["mean_iou"], 3),
           "hull_frac": round(float(hull), 3) if hull else None,
           "t_case": round(t_case, 1), "t_solve": round(t_solve, 2)}
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        img = montage(scat, assemble_image(scat, polys, pose, side), tex, S)
        cv2.imencode(".png", img)[1].tofile(str(save))
    return row


def load_tex(path: str, S: int = 760) -> np.ndarray:
    if path:
        t = J.read_color(path)
        if t is not None:
            return cv2.resize(t, (S, S)) if t.shape[0] != S else t
    from gen_hard_cases import draw_letter
    np.random.seed(0)
    return np.clip(draw_letter(S, S), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="v2", choices=["v1", "v2"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--image", default="")
    ap.add_argument("--size", type=int, default=760)
    ap.add_argument("--v1-len-tol", type=float, default=None,
                    help="v1 的等长容差（px）；不给就按噪声自动取 2+3σ")
    ap.add_argument("--v1-tex-tol", type=float, default=55.0)
    ap.add_argument("--sweep", action="store_true", help="v1 vs v2 在 σ=0/1.5/3 全表")
    ap.add_argument("--out", default=str(PROJ / "svg_tool" / "test_out" / "bench"))
    args = ap.parse_args()

    S = args.size
    tex = load_tex(args.image, S)
    outdir = pathlib.Path(args.out)

    jobs = []
    if args.sweep:
        for ns in (0.0, 1.5, 3.0):
            for sol in ("v1", "v2"):
                jobs.append((sol, args.n, ns))
    else:
        jobs.append((args.solver, args.n, args.noise))

    rows = []
    print(f"{'solver':<7}{'n':>4}{'σ':>6}{'placed':>9}{'拼对':>7}{'拼对率':>8}"
          f"{'meanIoU':>9}{'hull':>7}{'t_solve':>9}")
    print("-" * 66)
    for sol, n, ns in jobs:
        lt = args.v1_len_tol if args.v1_len_tol is not None else round(2.0 + 3.0 * ns, 2)
        tag = outdir / f"{sol}_n{n}_s{ns:g}.png"
        r = run_one(sol, n, ns, args.seed, tex, S, lt, args.v1_tex_tol, tag)
        rows.append(r)
        print(f"{sol:<7}{r['n']:>4}{ns:>6.1f}{r['placed']:>9}{r['correct']:>7}"
              f"{r['rate']:>7.1f}%{r['mean_iou']:>9.3f}"
              f"{str(r['hull_frac']):>7}{r['t_solve']:>9.2f}")

    import json
    (outdir / "bench.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n对比图 / 结果 → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
