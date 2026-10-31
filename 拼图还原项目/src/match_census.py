"""统计跨碎片的等长边（边长匹配普查）。

用法: python src/match_census.py --pieces out3/pieces.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out3/pieces.json")
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    edges = []
    for d in data["pieces"]:
        for i, e in enumerate(d["edges"]):
            edges.append((int(d["id"]), i, float(e["length"])))
    print(f"共 {len(edges)} 条边")

    pairs = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edges[i][0] == edges[j][0]:
                continue
            La, Lb = edges[i][2], edges[j][2]
            diff = abs(La - Lb) / max(La, Lb)
            pairs.append((diff, edges[i], edges[j]))
    pairs.sort()

    for tol in (0.01, 0.02, 0.03, 0.05, 0.08, 0.10):
        k = sum(1 for p in pairs if p[0] <= tol)
        print(f"容差 {tol*100:4.0f}%: {k:3d} 对")

    print("\n最接近的 25 对：")
    for diff, a, b in pairs[:25]:
        print(f"  #{a[0]}e{a[1]}({a[2]:.0f})  x  #{b[0]}e{b[1]}({b[2]:.0f})   Δ{diff*100:5.1f}%")

    # 每块有多少条边能找到等长伙伴（容差 3%）
    adj = {}
    for diff, a, b in pairs:
        if diff <= 0.03:
            adj.setdefault(a[0], []).append((a[1], b[0], b[1], diff))
            adj.setdefault(b[0], []).append((b[1], a[0], a[1], diff))
    print("\n3% 容差下各块可匹配的边数:")
    for pid in sorted({e[0] for e in edges}):
        lst = adj.get(pid, [])
        print(f"  #{pid}: {len(lst)} 条  " + "  ".join(f"e{e[0]}→#{e[1]}e{e[2]}" for e in lst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
