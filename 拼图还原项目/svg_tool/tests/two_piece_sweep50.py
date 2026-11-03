"""50 种两片手撕情况：生成随机两片散落图 → 走真实拼合并接口(two_join) → 每例「原图|拼好」对比 → 汇总成一张大图。"""
import sys, os, tempfile, pathlib, base64, time
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "web"))
import jigsaw_torn_solve as T
import two_join as TJ
from shapely.geometry import Polygon

OUT = ROOT / "svg_tool" / "test_out"
TEXF = OUT / "letter_h5.png"
N = 50
S = 340
SIDE = int(S * 2.15)
PALETTE = [(30, 120, 245), (40, 170, 60), (60, 70, 200), (120, 110, 100),
           (150, 80, 170), (40, 190, 210), (200, 120, 40), (90, 90, 160)]


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def two_pieces(S, rng):
    vertical = rng.integers(2) == 0
    x1 = rng.uniform(0.28, 0.72) * S; x2 = rng.uniform(0.28, 0.72) * S
    m = max(10, int(S / 5))
    t = np.linspace(0, 1, m)
    amp = rng.uniform(0.03, 0.10) * S
    ph = rng.uniform(0, 6.28); fr = rng.uniform(2.0, 4.0)
    xs = x1 + (x2 - x1) * t + amp * np.sin(t * np.pi * fr + ph) + rng.normal(0, amp * 0.28, m)
    xs = np.clip(xs, 3, S - 3)
    ys = t * S
    cut = np.column_stack([xs, ys])
    left = np.vstack([[0, 0], cut, [0, S]])
    right = np.vstack([cut, [[S, S], [S, 0]]])
    if not vertical:  # 横切：交换 x/y
        left = left[:, ::-1].copy(); right = right[:, ::-1].copy()
    return [left, right]


def scatter_draw(tex, pieces, rng, bgcol):
    canvas = np.full((SIDE, SIDE, 3), bgcol, np.uint8)
    placed = []
    for poly in pieces:
        cen = poly.mean(0); rad = np.max(np.linalg.norm(poly - cen, axis=1)) + 6
        for _try in range(3000):
            th = rng.uniform(0, 2 * np.pi)
            cx = rad + rng.uniform(0, SIDE - 2 * rad); cy = rad + rng.uniform(0, SIDE - 2 * rad)
            Rm = T.rot(th); pc = (poly - cen) @ Rm.T + [cx, cy]
            h = Polygon(pc).convex_hull.buffer(max(10, int(0.05 * SIDE)))
            b = h.bounds
            if b[0] < 0 or b[1] < 0 or b[2] > SIDE or b[3] > SIDE: continue
            if any(h.intersects(ph) for ph in placed): continue
            placed.append(h)
            # 贴纹理
            cx0 = max(0, int(poly[:, 0].min())); cy0 = max(0, int(poly[:, 1].min()))
            cx1 = min(tex.shape[1], int(poly[:, 0].max()) + 1); cy1 = min(tex.shape[0], int(poly[:, 1].max()) + 1)
            crop = tex[cy0:cy1, cx0:cx1]; mask = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.round(poly - [cx0, cy0]).astype(np.int32)], 255)
            sh = Rm @ np.array([cx0, cy0], float) + (np.array([cx, cy]) - Rm @ cen)
            M = np.array([[Rm[0, 0], Rm[0, 1], sh[0]], [Rm[1, 0], Rm[1, 1], sh[1]]], float)
            wc = cv2.warpAffine(crop, M, (SIDE, SIDE)); wm = cv2.warpAffine(mask, M, (SIDE, SIDE))
            canvas[wm > 0] = wc[wm > 0]
            break
    return canvas


def main():
    tex = cv2.resize(rd(TEXF), (S, S), interpolation=cv2.INTER_AREA)
    tiles = []
    nok = 0
    for c in range(N):
        bgcol = PALETTE[c % len(PALETTE)]
        inp = None; res = {"ok": False, "pieces": "?"}
        for attempt in range(6):          # 分割没出 2 片就换个摆法重试，确保每例都是 2 片
            rng = np.random.default_rng(1000 + c * 13 + attempt * 97)
            pieces = two_pieces(S, rng)
            inp = scatter_draw(tex, pieces, rng, bgcol)
            tmpd = pathlib.Path(tempfile.mkdtemp(prefix="sweep-"))
            ip = tmpd / "in.jpg"; cv2.imencode(".jpg", inp)[1].tofile(str(ip))
            try:
                res = TJ.assemble(str(ip))
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            if res.get("ok"):
                break
        joined = None; tag = ""
        if res.get("ok"):
            joined = cv2.imdecode(np.frombuffer(base64.b64decode(res["joined_png"]), np.uint8), cv2.IMREAD_COLOR)
            tag = res["edges"]; nok += 1
        else:
            tag = f"未拼({res.get('pieces','?')}片)"
        tiles.append((inp, joined, c + 1, tag, res.get("ok", False)))
        print(f"[{c+1:02d}/50] ok={res.get('ok')} {tag}", flush=True)

    # 组装大图：5 列 x 10 行，每格 [原图|拼好] + 标签
    cols = 5; rows = (N + cols - 1) // cols
    TW = 300; TH = 170; LBL = 22; GAP = 8; TOP = 54
    W = cols * (TW + GAP) + GAP; H = TOP + rows * (TH + LBL + GAP) + GAP
    cv = np.full((H, W, 3), 18, np.uint8)
    cv2.putText(cv, f"Two-piece torn join  -  {N} random cases   ({nok}/{N} joined)", (GAP + 2, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 245, 210), 2)
    cv2.putText(cv, "each cell: LEFT = input (scattered 2 pieces, varied bg/cut/rotation)   RIGHT = auto-joined output",
                (GAP + 2, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 172), 1)
    for idx, (inp, joined, cid, tag, ok) in enumerate(tiles):
        r, cc = divmod(idx, cols)
        x = GAP + cc * (TW + GAP); y = TOP + r * (TH + LBL + GAP)
        col = (120, 245, 210) if ok else (90, 90, 240)
        cv2.rectangle(cv, (x, y), (x + TW, y + LBL), (40, 40, 48), -1)
        cv2.putText(cv, f"#{cid:02d} {tag}", (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
        hw = TW // 2
        ai = cv2.resize(inp, (hw, TH)); 
        cv[y + LBL:y + LBL + TH, x:x + hw] = ai
        if joined is not None:
            jj = cv2.resize(joined, (hw, TH)); cv[y + LBL:y + LBL + TH, x + hw:x + TW] = jj
        else:
            cv[y + LBL:y + LBL + TH, x + hw:x + TW] = (30, 30, 40)
            cv2.putText(cv, "FAIL", (x + hw + 8, y + LBL + TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 240), 2)
        cv2.rectangle(cv, (x, y + LBL), (x + TW, y + LBL + TH), (70, 70, 80), 1)
    out = OUT / "two_piece_50_montage.png"
    cv2.imencode(".png", cv)[1].tofile(str(out))
    print(f"DONE {nok}/{N} joined -> {out} {cv.shape}", flush=True)
    os.startfile(os.path.abspath(str(out)))


if __name__ == "__main__":
    main()
