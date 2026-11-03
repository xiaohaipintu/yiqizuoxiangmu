"""给两片手撕件的每条边起名并标注（约定：最上方顶点起，环状编号；第k边=顶点k→k+1）。"""
import sys, json, pathlib, tempfile, os
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[2]
tmp = pathlib.Path(tempfile.gettempdir()) / "two"
out = ROOT / "svg_tool" / "test_out"


def rd(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def wr(p, im): cv2.imencode(".png", im)[1].tofile(str(p))


def start_topmost(poly):
    """把多边形重排成从最上方顶点(同高取最左)开始，保持原有绕向。"""
    poly = np.asarray(poly, float)
    idx = np.lexsort((poly[:, 0], poly[:, 1]))[0]  # 先按y再按x → 最上、同高最左
    return np.roll(poly, -idx, axis=0)


def main():
    doc = json.loads((tmp / "seg" / "pieces.json").read_text(encoding="utf-8"))
    pieces = doc["pieces"]
    # 排序：上→下、左→右 命名 A、B
    pieces.sort(key=lambda p: (round(p["centroid"][1] / 200), p["centroid"][0]))
    names = ["A", "B", "C", "D"]
    orig = rd(tmp / "two.jpg")
    H, W = orig.shape[:2]
    scale = 1600.0 / max(H, W)
    canvas = cv2.resize(orig, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    colors = [(255, 120, 40), (60, 60, 255), (40, 200, 40), (200, 60, 200)]

    table = []
    for pi, p in enumerate(pieces):
        name = names[pi]
        poly = start_topmost(np.array(p["polygon"], float)) * scale
        col = colors[pi % len(colors)]
        cen = poly.mean(0)
        n = len(poly)
        # 半透明填充 + 描边
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.round(poly).astype(np.int32)], col)
        canvas = cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0)
        cv2.polylines(canvas, [np.round(poly).astype(np.int32)], True, col, 2)
        # 片名
        cv2.putText(canvas, name, (int(cen[0]) - 12, int(cen[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 5)
        cv2.putText(canvas, name, (int(cen[0]) - 12, int(cen[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)
        # 每条边：顶点 k -> k+1，命名 name+(k+1)
        for k in range(n):
            a = poly[k]; b = poly[(k + 1) % n]
            mid = (a + b) / 2
            L = float(np.linalg.norm((b - a) / scale))
            d = b - a; Ln = np.linalg.norm(d) or 1.0
            nx, ny = -d[1] / Ln, d[0] / Ln
            if np.dot([nx, ny], mid - cen) < 0:
                nx, ny = -nx, -ny
            pos = mid + np.array([nx, ny]) * 26
            lab = f"{name}{k + 1}"
            cv2.line(canvas, (int(mid[0]), int(mid[1])), (int(pos[0]), int(pos[1])), col, 1)
            cv2.putText(canvas, lab, (int(pos[0]) - 12, int(pos[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(canvas, lab, (int(pos[0]) - 12, int(pos[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            table.append((lab, round(L, 1)))

    wr(out / "two_edges_named.png", canvas)
    print("边名 + 长度(px):")
    for lab, L in table:
        print(f"  {lab:>4}  {L}")
    print("saved", out / "two_edges_named.png", canvas.shape)
    os.startfile(os.path.abspath(str(out / "two_edges_named.png")))


if __name__ == "__main__":
    main()
