"""给每个碎片的每条边取名字并画出来（诊断用）。

命名规则：`#块号-边号`，边号按多边形顶点顺序 A、B、C……循环。
标签画在边的中点、朝块内偏移一点，白底描边，黑底/白纸都看得清。
同时打印一张表：边名、长度、起点、终点、方向角、中点。

用法：
    python 纹理拼合/tools/label_edges.py --image 手撕拼图.jpg --out 纹理拼合/手撕拼图
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "拼图还原项目" / "src"))
sys.path.insert(0, str(ROOT / "纹理拼合" / "tools"))

from irregular_cut import to_pieces_json  # noqa: E402
from handmade_puzzle import imread_u, imwrite_u, segment  # noqa: E402

LETTERS = "ABCDEFGHIJ"
# 每条边一个颜色（BGR）
COLORS = [(60, 90, 240), (60, 200, 100), (240, 140, 60), (220, 80, 220),
          (60, 220, 230), (200, 70, 180), (120, 200, 60), (250, 190, 60)]


def put_label(img, text, center, scale, fg, pad_inward=None):
    """白描边 + 彩色填充的文字，居中在 center。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(2, int(round(scale * 4)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = int(center[0] - tw / 2)
    y = int(center[1] + th / 2)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick + 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, fg, thick, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="给每块碎片的每条边编号命名")
    ap.add_argument("--image", default="手撕拼图.jpg")
    ap.add_argument("--out", default="纹理拼合/手撕拼图")
    ap.add_argument("--th", type=int, default=100)
    ap.add_argument("--min-area", type=float, default=20000)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--name", default="6_碎片边编号.jpg")
    args = ap.parse_args()

    src = imread_u(Path(args.image))
    cells, _ = segment(src, args.th, args.min_area, args.eps)
    pieces = to_pieces_json(cells)["pieces"]

    out = src.copy()
    # 半透明压暗一点，让彩色标注更突出
    out = (out * 0.45 + 255 * 0.0).astype(np.uint8)

    rows = []
    for p in pieces:
        pid = int(p["id"])
        poly = np.asarray(p["polygon"], float)
        n = len(poly)
        centroid = poly.mean(axis=0)
        for i, e in enumerate(p["edges"]):
            name = "#%d-%s" % (pid, LETTERS[i])
            a = np.asarray(e["a"], float)
            b = np.asarray(e["b"], float)
            col = COLORS[i % len(COLORS)]
            # 边本身加粗画出来
            cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     col, 5, cv2.LINE_AA)
            mid = (a + b) / 2
            inward = centroid - mid
            u = inward / (np.linalg.norm(inward) + 1e-9)
            pos = mid + u * 34
            put_label(out, name, pos, 0.95, col)
            rows.append({
                "edge": name,
                "piece": pid,
                "index": i,
                "length": round(float(e["length"]), 1),
                "angle": round(float(e["angle"]), 1),
                "a": [round(float(v), 1) for v in e["a"]],
                "b": [round(float(v), 1) for v in e["b"]],
                "mid": [round(float(v), 1) for v in e["mid"]],
            })

    # 顶部加一行说明
    cv2.putText(out, "edge names: #piece-letter  (A,B,C... in vertex order)",
                (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(out, "edge names: #piece-letter  (A,B,C... in vertex order)",
                (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 3, cv2.LINE_AA)

    outdir = Path(args.out)
    imwrite_u(outdir / args.name, out)
    (outdir / "edges.json").write_text(
        json.dumps({"pieces": len(pieces), "edges": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("边清单（%d 块，共 %d 条边）" % (len(pieces), len(rows)))
    print("%-8s %8s %8s   %-22s %-22s" % ("边名", "长度px", "方向°", "起点", "终点"))
    for r in rows:
        print("%-8s %8.1f %8.1f   (%7.1f,%7.1f)        (%7.1f,%7.1f)"
              % (r["edge"], r["length"], r["angle"],
                 r["a"][0], r["a"][1], r["b"][0], r["b"][1]))
    print()
    print("图 →", outdir / args.name)
    print("数据 →", outdir / "edges.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
