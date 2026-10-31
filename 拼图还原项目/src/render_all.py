"""把枚举出来的所有拼法铺成一张大图。

只影响"排版顺序"、不影响"是否存在"：按拼接结果的外轮廓紧致度（空穴少、轮廓短）排序，
最像完整图形的排在前面，方便肉眼先看前几十个。

用法:
    python src/render_all.py --assemblies out/enum5/assemblies.json --out out
输出:
    out/06_all_assemblies.png   全部拼法的大图（自适应格子大小）
    out/06_preview.png          缩小版预览
    out/07_top_compact.png      最紧致的前 48 种，大图细看
    out/assembly_stats.json     每种拼法的空穴率/外轮廓长度等指标
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
MAX_W = 6200
MAX_PIX = 62_000_000


def assembly_stats(polys: list[np.ndarray]) -> dict:
    pts = np.vstack(polys)
    hull = cv2.convexHull(pts.astype(np.float32))
    hull_area = abs(float(cv2.contourArea(hull)))
    sum_area = sum(G.polygon_area(p) for p in polys)
    holes = max(0.0, hull_area - sum_area)
    return {
        "sum_area": sum_area,
        "hull_area": hull_area,
        "holes": holes,
        "holes_pct": holes / max(sum_area, 1e-9) * 100.0,
        "hull_perimeter": float(cv2.arcLength(hull, True)),
    }


def fit_polys(polys: list[np.ndarray], box: float) -> tuple[list[np.ndarray], float]:
    pts = np.vstack(polys)
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    s = (box - 10.0) / max(w, h)
    mn = np.array([pts[:, 0].min(), pts[:, 1].min()])
    return [(p - mn) * s + 5.0 for p in polys], s


def draw_tile(canvas: np.ndarray, polys: list[np.ndarray], ids: list[int], x: int, y: int,
              box: int, label: str | None, label_color=(40, 40, 40)):
    scaled, _ = fit_polys(polys, box)
    for p, pid in zip(scaled, ids):
        pts = (p + [x, y]).astype(np.int32)
        cv2.fillPoly(canvas, [pts], PALETTE[(pid - 1) % len(PALETTE)])
        cv2.polylines(canvas, [pts], True, (60, 60, 60), 1, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (x, y + box + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, label_color, 1, cv2.LINE_AA)


def pick_tile(n: int, pad: int = 10) -> tuple[int, int]:
    """根据数量选格子边长，保证整图不超过像素预算。"""
    for tile in (120, 100, 88, 76, 64, 56, 48, 42, 36, 30):
        cols = max(1, int((MAX_W - 20) // (tile + pad)))
        rows = math.ceil(n / cols)
        wpx = 20 + cols * (tile + pad)
        hpx = 150 + rows * (tile + pad + 16)
        if wpx * hpx <= MAX_PIX:
            return tile, cols
    return 30, max(1, int((MAX_W - 20) // 40))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemblies", default="out/enum5/assemblies.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--top", type=int, default=48)
    ap.add_argument("--len-tol", default="5%")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    sols = json.loads(pathlib.Path(args.assemblies).read_text(encoding="utf-8"))
    n = len(sols)

    # 统计 + 排序（紧凑的在前）
    stats = []
    for s in sols:
        polys = [np.array(p["polygon_world"], dtype=float) for p in s["pieces"]]
        st = assembly_stats(polys)
        st["index"] = s["index"]
        stats.append(st)
    order = sorted(range(n), key=lambda i: (stats[i]["holes_pct"], stats[i]["hull_perimeter"]))
    (outdir / "assembly_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    tile, cols = pick_tile(n)
    pad = 10
    rows = math.ceil(n / cols)
    W = 20 + cols * (tile + pad)
    H = 130 + rows * (tile + pad + 16)
    canvas = np.full((H, W, 3), 255, np.uint8)

    cv2.putText(canvas, f"Jigsaw reassembly - all edge-length-compatible layouts: {n}", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"rule: edge length match only (tol {args.len_tol}), rotation + flip allowed; "
                        f"order: tightest outline first", (16, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, "colors = the 7 fragments (same color scheme as the photo)", (16, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    # 颜色图例
    lx = 16
    for k in range(7):
        cv2.rectangle(canvas, (lx, 108), (lx + 16, 124), PALETTE[k], -1)
        cv2.putText(canvas, f"#{k+1}", (lx + 20, 121), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        lx += 54

    shown = 0
    for pos, idx in enumerate(order):
        s = sols[idx]
        polys = [np.array(p["polygon_world"], dtype=float) for p in s["pieces"]]
        ids = [p["id"] for p in s["pieces"]]
        r, c = divmod(pos, cols)
        x = 10 + c * (tile + pad)
        y = 140 + r * (tile + pad + 16)
        draw_tile(canvas, polys, ids, x, y, tile, label=f"{idx+1}")
        shown += 1

    cv2.imwrite(str(outdir / "06_all_assemblies.png"), canvas)
    print(f"写出 {outdir / '06_all_assemblies.png'}  {canvas.shape[1]}x{canvas.shape[0]}  ({shown} 个拼法)")

    preview = cv2.resize(canvas, (1600, int(canvas.shape[0] * 1600 / canvas.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(outdir / "06_preview.png"), preview)
    print(f"写出 {outdir / '06_preview.png'}  {preview.shape[1]}x{preview.shape[0]}")

    # 前 K 个细看
    top = order[: args.top]
    tcols = 8
    trot = math.ceil(len(top) / tcols)
    tbox = 190
    tW = 20 + tcols * tbox
    tH = 130 + trot * (tbox + 34)
    det = np.full((tH, tW, 3), 255, np.uint8)
    cv2.putText(det, f"Top {len(top)} tightest layouts (same edge-length rule)", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    for pos, idx in enumerate(top):
        s = sols[idx]
        polys = [np.array(p["polygon_world"], dtype=float) for p in s["pieces"]]
        ids = [p["id"] for p in s["pieces"]]
        r, c = divmod(pos, tcols)
        x = 10 + c * tbox
        y = 130 + r * (tbox + 34)
        draw_tile(det, polys, ids, x, y, tbox - 12, label=None)
        cv2.putText(det, f"#{idx+1}  holes {stats[idx]['holes_pct']:.1f}%", (x, y + tbox + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(outdir / "07_top_compact.png"), det)
    print(f"写出 {outdir / '07_top_compact.png'}  {det.shape[1]}x{det.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
