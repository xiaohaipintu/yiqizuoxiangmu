"""看 T 型接缝搜索的结果里，有没有「A2 与 B4 相邻」以及更像矩形的解。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402

JOB = Path("C:/Users/Administrator/AppData/Local/Temp/jigsaw-jobs/fine2")
pieces = json.loads((JOB / "pieces.json").read_text(encoding="utf-8"))["pieces"]
names, starts = {}, {}
for i, p in enumerate(pieces):
    pid = int(p["id"])
    names[pid] = P.piece_name(i)
    starts[pid] = P.edge_start_index(np.asarray(p["polygon"], float))

doc = json.loads((JOB / "split_solutions.json").read_text(encoding="utf-8"))
sols = doc["solutions"]
print(f"T 型接缝候选 {doc['n']} 个\n")


def touches(entry, ea, eb, share=0.25):
    """ea、eb 两条边是否有共线重叠段（允许部分贴合）。"""
    for j in P.find_joints(_labeled(entry), share=share):
        if {j["a"], j["b"]} == {ea, eb}:
            return j["share"]
    return 0.0


def _labeled(entry):
    polys = entry["polys"]
    ids = [int(x[0]) for x in entry["placed"]]
    return [(names[pid], np.asarray(poly, float), starts[pid])
            for pid, poly in zip(ids, polys)]


rows = []
for i, e in enumerate(sols):
    m = P.solution_metrics(e, names, starts, share=0.25)
    if m is None:
        continue
    a2b4 = touches(e, "A2", "B4")
    rows.append({"i": i, "m": m, "a2b4": a2b4})

rows.sort(key=lambda r: r["m"]["score"])
print(f"{'候选':>6}{'分数':>8}{'长度差%':>9}{'凸包%':>8}{'矩形%':>8}{'轮廓点':>7}{'A2↔B4':>9}")
for r in rows[:10]:
    m = r["m"]
    print(f"{'#' + str(r['i']):>6}{m['score']:>8}{m['len_diff_pct']:>9}{m['hull_gap_pct']:>8}"
          f"{m.get('rect_fill_pct', 0):>8}{m.get('outline_vertices', 0):>7}{r['a2b4']:>9}")

with_a2b4 = [r for r in rows if r["a2b4"] > 0]
print(f"\n含 A2↔B4 相邻的候选：{len(with_a2b4)} 个")
if with_a2b4:
    b = max(with_a2b4, key=lambda r: r["a2b4"])
    print(f"  贴合最紧的：#{b['i']}  A2↔B4 贴合 {b['a2b4']}%  矩形 {b['m'].get('rect_fill_pct')}%  "
          f"长度差 {b['m']['len_diff_pct']}%")

best_rect = max(rows, key=lambda r: r["m"].get("rect_fill_pct", 0))
print(f"\n矩形贴合度最高：#{best_rect['i']}  {best_rect['m'].get('rect_fill_pct')}%  "
      f"轮廓 {best_rect['m'].get('outline_vertices')} 点  长度差 {best_rect['m']['len_diff_pct']}%  "
      f"A2↔B4 {best_rect['a2b4']}%")
