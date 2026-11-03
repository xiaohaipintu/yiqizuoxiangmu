"""手撕拼图求解 demo：生成共享撕痕的散落碎片 → 求解 → 渲染「输入|拼好」对比图。"""
import sys, pathlib, time
import numpy as np
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import jigsaw_torn_solve as T

BASE = pathlib.Path("拼图还原项目/svg_tool/test_out")
TEX = BASE / "letter_h5.png"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def render(cells, transforms, side, tex):
    canvas = np.zeros((side, side, 3), np.uint8)
    H, W = tex.shape[:2]
    for k, tr in enumerate(transforms):
        if tr is None:
            continue
        R, t = tr
        cell = np.asarray(cells[k], float)
        cx0 = max(0, int(np.floor(cell[:, 0].min()))); cy0 = max(0, int(np.floor(cell[:, 1].min())))
        cx1 = min(W, int(np.ceil(cell[:, 0].max())) + 1); cy1 = min(H, int(np.ceil(cell[:, 1].max())) + 1)
        crop = tex[cy0:cy1, cx0:cx1]
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(cell - [cx0, cy0]).astype(np.int32)], 255)
        shift = R @ np.array([cx0, cy0], float) + t
        M = np.array([[R[0, 0], R[0, 1], shift[0]], [R[1, 0], R[1, 1], shift[1]]], float)
        wc = cv2.warpAffine(crop, M, (side, side), flags=cv2.INTER_LINEAR)
        wm = cv2.warpAffine(mask, M, (side, side), flags=cv2.INTER_NEAREST)
        canvas[wm > 0] = wc[wm > 0]
    return canvas


def main():
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    tex = rd(TEX); S = tex.shape[0]
    cells = T.torn_tessellation(G, S, seed)
    scat = T.scatter(cells, S, seed)
    n = len(cells)

    # 每片的 单元→散落 刚体变换
    Sxf = [T.procrustes(cells[k], scat[k]) for k in range(n)]
    in_side = int(S * 1.9)
    inp = render(cells, Sxf, in_side, tex)

    t0 = time.time()
    pose, placed, pp = T.solve_v2(scat, min_run=10, seed=seed)
    dt = time.time() - t0
    m = T.evaluate(pp)

    # 总变换 = 求解位姿(散落→世界) ∘ 散落变换(单元→散落)
    totals = []
    for k in range(n):
        if pose[k] is None:
            totals.append(None); continue
        Rs, ts = Sxf[k]; Rp, tp = pose[k]
        totals.append((Rp @ Rs, Rp @ ts + tp))
    worlds = np.vstack([cells[k] @ totals[k][0].T + totals[k][1] for k in range(n) if totals[k] is not None])
    x0 = int(np.floor(worlds[:, 0].min())) - 2; y0 = int(np.floor(worlds[:, 1].min())) - 2
    side = int(np.ceil(max(worlds[:, 0].max() - x0, worlds[:, 1].max() - y0))) + 4
    totals_shift = [None if tr is None else (tr[0], tr[1] - np.array([x0, y0])) for tr in totals]
    out = render(cells, totals_shift, side, tex)

    wr(BASE / "torn_input.png", inp)
    wr(BASE / "torn_assembled.png", out)
    print(f"G={G} seed={seed} pieces={n} placed={placed} overlap%={m.get('overlap_pct'):.3f} "
          f"hull={m.get('hull_frac'):.3f} solve={dt:.2f}s")

    # 侧边对比
    SZ = 780
    a = cv2.resize(inp, (SZ, SZ)); b = cv2.resize(out, (SZ, SZ))
    TOP = 66; GAP = 16
    cv = np.full((TOP + SZ + GAP * 2, SZ * 2 + GAP * 3, 3), 20, np.uint8)

    def txt(s, o, sc, c, th=1): cv2.putText(cv, s, o, cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)
    txt(f"Hand-torn jigsaw  {n} pieces  |  shared torn seams + fuzzy edges", (GAP + 2, 26), 0.6, (120, 245, 210), 2)
    txt(f"LEFT: scattered input       RIGHT: reassembled by solver  (placed {placed}/{n}, overlap {m.get('overlap_pct'):.2f}%)",
        (GAP + 2, 52), 0.5, (165, 165, 175), 1)
    cv[TOP:TOP + SZ, GAP:GAP + SZ] = a; cv2.rectangle(cv, (GAP, TOP), (GAP + SZ, TOP + SZ), (70, 70, 80), 1)
    x = GAP * 2 + SZ; cv[TOP:TOP + SZ, x:x + SZ] = b; cv2.rectangle(cv, (x, TOP), (x + SZ, TOP + SZ), (70, 70, 80), 1)
    txt("INPUT (torn, scattered)", (GAP + 8, TOP + 24), 0.6, (235, 235, 235), 1)
    txt("REASSEMBLED", (x + 8, TOP + 24), 0.6, (120, 245, 210), 1)
    out_path = BASE / "torn_solve_compare.png"
    wr(out_path, cv)
    print("saved", out_path)
    import os
    os.startfile(os.path.abspath(str(out_path)))


if __name__ == "__main__":
    main()
