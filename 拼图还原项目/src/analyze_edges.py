"""诊断：统计碎片之间的边长可匹配对，用来选容差。"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

data = json.loads(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/pieces.json").read_text(encoding="utf-8"))
pieces = data["pieces"]

edges = []
for p in pieces:
    for e in p["edges"]:
        edges.append((p["id"], e["index"], float(e["length"]), float(e["angle"]) % 180.0))

print(f"共 {len(edges)} 条边")
for tol in (0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.10, 0.15):
    pairs = 0
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            a, b = edges[i], edges[j]
            if a[0] == b[0]:
                continue
            if abs(a[2] - b[2]) / max(a[2], b[2]) <= tol:
                pairs += 1
    print(f"  容差 {tol*100:>4.0f}%: 可匹配边对 {pairs:>4} 对")

print("\n3% 容差下的边对明细（只列跨碎片的）：")
for i in range(len(edges)):
    for j in range(i + 1, len(edges)):
        a, b = edges[i], edges[j]
        if a[0] == b[0]:
            continue
        if abs(a[2] - b[2]) / max(a[2], b[2]) <= 0.03:
            print(f"  #{a[0]}e{a[1]} {a[2]:>6.1f}px  <->  #{b[0]}e{b[1]} {b[2]:>6.1f}px   Δ={abs(a[2]-b[2]):.1f}px")
