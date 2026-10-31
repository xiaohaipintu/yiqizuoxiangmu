"""穷举所有拼法（只按边长匹配，可选关闭镜像/开启透视缩放补偿）。

规则：
    两块碎片要能贴合，必须有一条长度相等的边。
    拍照透视会让同一张纸上的碎片大小差 3~7%，所以可选「缩放补偿」：
    贴合时允许把待放的碎片缩放 s=La/Lb（默认不启用，启用后 s 必须在 ±5% 内）。
    允许任意角度旋转；镜像默认开启，可用 --no-mirror 关闭。

一个完整拼法 = 所有碎片都用上并连成一体。
同一拼法会被不同拼序得到，也会因误差产生像素级变体，因此两级去重：
  1) 精确签名（相对刚体变换，整体平移/旋转/镜像等价视为同一种）
  2) 容差同构（每块相对位置 4px、相对角度 4°、缩放 2% 内视为同一种）

用法:
    python src/enumerate.py --pieces out/pieces.json --out out --len-tol 0.05
    python src/enumerate.py --pieces out2/pieces.json --out out2 --no-mirror --allow-scale
输出:
    out/assemblies.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as G  # noqa: E402

sys.setrecursionlimit(10000)


class Piece:
    def __init__(self, d: dict):
        self.id = int(d["id"])
        self.poly = np.array(d["polygon"], dtype=float)
        self.area = float(d["area"])
        self.edges = list(d["edges"])
        self.n = len(self.poly)


def affine(deg: float, t, mirror: bool, scale: float = 1.0) -> np.ndarray:
    S = np.array([[-1.0, 0.0], [0.0, 1.0]]) if mirror else np.eye(2)
    M = np.eye(3)
    M[:2, :2] = (G.rotation(deg) @ S) * scale
    M[:2, 2] = np.asarray(t, float)
    return M


def world(piece: Piece, deg: float, t, mirror: bool, scale: float = 1.0) -> np.ndarray:
    M = affine(deg, t, mirror, scale)[:2, :]
    return (M[:, :2] @ piece.poly.T).T + M[:, 2]


def placement(piece: Piece, edge_idx: int, ta, tb, mirror: bool, flip: bool,
              allow_scale: bool, max_scale_dev: float):
    """把 piece 的第 edge_idx 条边贴到目标边 (ta,tb)，返回 (角度, 平移, 缩放)。"""
    e = piece.edges[edge_idx]
    a = np.array(e["a"], float)
    b = np.array(e["b"], float)
    Lb = float(e["length"])
    if mirror:
        a = np.array([-a[0], a[1]])
        b = np.array([-b[0], b[1]])
    ds = b - a
    ds /= np.linalg.norm(ds) + 1e-12
    tab = np.asarray(tb, float) - np.asarray(ta, float)
    La = float(np.linalg.norm(tab))
    dt = tab / (La + 1e-12)

    s = 1.0
    if allow_scale:
        s = La / max(Lb, 1e-9)
        if abs(s - 1.0) > max_scale_dev:
            return None

    if flip:
        want, src, dst = dt, a, np.asarray(ta, float)
    else:
        want, src, dst = -dt, a, np.asarray(tb, float)
    ang = math.degrees(math.atan2(want[1], want[0]) - math.atan2(ds[1], ds[0])) % 360.0
    t = dst - (G.rotation(ang) @ src) * s
    return ang, t, s


def rel_matrices(placed) -> list:
    """相对参考碎片（id 最小）的变换，整体平移/旋转/镜像下不变。"""
    ref = min(placed, key=lambda r: r[0])
    inv = np.linalg.inv(affine(ref[1], (ref[2], ref[3]), ref[4], ref[5]))
    out = []
    for pid, deg, tx, ty, mir, s in sorted(placed, key=lambda r: r[0]):
        M = (inv @ affine(deg, (tx, ty), mir, s))[:2, :]
        ang = math.degrees(math.atan2(M[1, 0], M[0, 0])) % 360.0
        out.append((pid, ang, float(M[0, 2]), float(M[1, 2]), mir, float(s)))
    return out


def exact_key(placed) -> tuple:
    return tuple((p, round(a, 1), round(x, 1), round(y, 1), int(m), round(s, 2))
                 for p, a, x, y, m, s in rel_matrices(placed))


def coarse_key(rel: list, dp: float, da: float) -> tuple:
    return tuple(sorted((p, int(round(a / da)), int(round(x / dp)), int(round(y / dp)), int(m),
                         int(round(s * 100 / 2.0)))
                        for p, a, x, y, m, s in rel))


class Dedupe:
    """容差同构去重：粗粒度哈希分桶，桶内按相对位姿逐一比对。"""

    def __init__(self, dp: float = 4.0, da: float = 4.0, ds: float = 0.02):
        self.dp, self.da, self.ds = dp, da, ds
        self.cell_p = max(8.0, dp * 3)
        self.cell_a = max(8.0, da * 3)
        self.buckets = {}

    def _close(self, A: list, B: list) -> bool:
        if len(A) != len(B):
            return False
        for a, b in zip(A, B):
            if a[0] != b[0] or int(a[4]) != int(b[4]):
                return False
            if abs((a[1] - b[1] + 180.0) % 360.0 - 180.0) > self.da:
                return False
            if math.hypot(a[2] - b[2], a[3] - b[3]) > self.dp:
                return False
            if abs(a[5] - b[5]) > self.ds:
                return False
        return True

    def add(self, rel: list) -> bool:
        keys = [coarse_key(rel, self.cell_p, self.cell_a)]
        shifted = [(p, a, x - self.cell_p / 2, y - self.cell_p / 2, m, s) for p, a, x, y, m, s in rel]
        keys.append(coarse_key(shifted, self.cell_p, self.cell_a))
        for k in keys:
            for rep in self.buckets.get(k, ()):
                if self._close(rep, rel):
                    return False
        self.buckets.setdefault(keys[0], []).append(rel)
        return True


class State:
    __slots__ = ("placed", "polys", "bboxes")

    def __init__(self, placed, polys):
        self.placed = placed
        self.polys = polys
        self.bboxes = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polys]


class Enumerator:
    def __init__(self, pieces, len_tol=0.05, overlap_tol=0.005, mirror=True,
                 allow_scale=False, max_scale_dev=0.05, verbose=True,
                 max_states=2_000_000, max_seconds=900.0, max_results=100_000):
        self.pieces = {p.id: p for p in pieces}
        self.ids = list(self.pieces)
        self.len_tol = len_tol
        self.overlap_tol = overlap_tol
        self.mirror = mirror
        self.allow_scale = allow_scale
        self.max_scale_dev = max_scale_dev
        self.verbose = verbose
        self.max_states = max_states
        self.max_seconds = max_seconds
        self.max_results = max_results
        self.log = []
        self.visited = [set() for _ in range(len(self.ids) + 1)]
        self.ded = Dedupe(dp=4.0, da=4.0, ds=0.02 if allow_scale else 1e-3)
        self.results = []
        self.states_expanded = 0
        self.cands_tested = 0
        self.truncated = False

    def _say(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg, flush=True)

    def initial(self, pid: int) -> State:
        p = self.pieces[pid]
        return State([(pid, 0.0, 0.0, 0.0, False, 1.0)], [world(p, 0.0, (0.0, 0.0), False, 1.0)])

    def expand(self, st: State):
        used = {r[0] for r in st.placed}
        out = []
        mirror_opts = (False, True) if self.mirror else (False,)
        for k, (pid, deg, tx, ty, mir, sc) in enumerate(st.placed):
            p = self.pieces[pid]
            pw = st.polys[k]
            cw = pw.mean(axis=0)
            for ei in range(p.n):
                aw, bw = pw[ei], pw[(ei + 1) % p.n]
                La = float(np.linalg.norm(bw - aw))
                if La < 1e-6:
                    continue
                u = (bw - aw) / La
                for qid in self.ids:
                    if qid in used:
                        continue
                    q = self.pieces[qid]
                    for fi in range(len(q.edges)):
                        Lb = float(q.edges[fi]["length"])
                        ratio = Lb / La
                        if self.allow_scale:
                            if not (1.0 / (1.0 + self.max_scale_dev) <= ratio <= 1.0 + self.max_scale_dev):
                                continue
                        elif abs(ratio - 1.0) > self.len_tol:
                            continue
                        for mirq in mirror_opts:
                            for flip in (False, True):
                                res = placement(q, fi, aw, bw, mirq, flip, self.allow_scale, self.max_scale_dev)
                                if res is None:
                                    continue
                                ang, t, s = res
                                wq = world(q, ang, t, mirq, s)
                                self.cands_tested += 1
                                cq = wq.mean(axis=0)
                                side_p = u[0] * (cw[1] - aw[1]) - u[1] * (cw[0] - aw[0])
                                side_q = u[0] * (cq[1] - aw[1]) - u[1] * (cq[0] - aw[0])
                                if side_p * side_q >= 0:
                                    continue
                                bq = (wq[:, 0].min(), wq[:, 1].min(), wq[:, 0].max(), wq[:, 1].max())
                                ov = 0.0
                                for j, sp in enumerate(st.polys):
                                    sb = st.bboxes[j]
                                    if bq[0] >= sb[2] or bq[2] <= sb[0] or bq[1] >= sb[3] or bq[3] <= sb[1]:
                                        continue
                                    ov += G.convex_overlap_area(wq, sp)
                                    if ov > self.overlap_tol * q.area:
                                        break
                                if ov > self.overlap_tol * q.area:
                                    continue
                                out.append(State(st.placed + [(qid, ang, float(t[0]), float(t[1]), mirq, s)],
                                                 st.polys + [wq]))
        return out

    def run(self):
        t0 = time.time()
        seed = max(self.ids, key=lambda i: self.pieces[i].area)
        st = self.initial(seed)
        self.visited[1].add(exact_key(st.placed))
        stack = [st]
        while stack:
            if time.time() - t0 > self.max_seconds or self.states_expanded > self.max_states:
                self.truncated = True
                self._say(f"!! 达到预算，枚举被截断（状态 {self.states_expanded}，用时 {time.time() - t0:.0f}s）")
                break
            st = stack.pop()
            n = len(st.placed)
            if n == len(self.pieces):
                rel = rel_matrices(st.placed)
                if self.ded.add(rel):
                    self.results.append(st)
                if len(self.results) >= self.max_results:
                    self.truncated = True
                    self._say("!! 结果数达到上限")
                    break
                continue
            self.states_expanded += 1
            for ch in self.expand(st):
                k = exact_key(ch.placed)
                if k in self.visited[len(ch.placed)]:
                    continue
                self.visited[len(ch.placed)].add(k)
                stack.append(ch)
            if self.states_expanded % 2000 == 0:
                self._say(
                    f"展开 {self.states_expanded} | 栈 {len(stack)} | 拼法 {len(self.results)} | "
                    f"候选 {self.cands_tested} | {time.time() - t0:.0f}s"
                )
        self._say(f"枚举结束：展开 {self.states_expanded}，候选 {self.cands_tested}，"
                  f"独立拼法 {len(self.results)}，用时 {time.time() - t0:.0f}s")
        return self.results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="out/pieces.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--len-tol", type=float, default=0.05)
    ap.add_argument("--overlap-tol", type=float, default=0.005)
    ap.add_argument("--no-mirror", dest="no_mirror", action="store_true")
    ap.add_argument("--allow-scale", dest="allow_scale", action="store_true")
    ap.add_argument("--max-scale-dev", type=float, default=0.05)
    ap.add_argument("--max-states", type=int, default=2_000_000)
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--max-results", type=int, default=100_000)
    args = ap.parse_args()

    data = json.loads(pathlib.Path(args.pieces).read_text(encoding="utf-8"))
    pieces = [Piece(d) for d in data["pieces"]]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    en = Enumerator(pieces, len_tol=args.len_tol, overlap_tol=args.overlap_tol,
                    mirror=not args.no_mirror, allow_scale=args.allow_scale,
                    max_scale_dev=args.max_scale_dev, max_states=args.max_states,
                    max_seconds=args.max_seconds, max_results=args.max_results)
    results = en.run()

    out = []
    for i, st in enumerate(results, start=1):
        pts = np.vstack(st.polys)
        off = np.array([pts[:, 0].min(), pts[:, 1].min()])
        out.append({
            "index": i,
            "pieces": [{
                "id": pid,
                "rotate_deg": float(deg % 360.0),
                "mirror": bool(mir),
                "scale": float(s),
                "translate": [float(tx - off[0]), float(ty - off[1])],
                "polygon_world": (w - off).tolist(),
            } for (pid, deg, tx, ty, mir, s), w in zip(st.placed, st.polys)],
        })

    (outdir / "assemblies.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (outdir / "enumerate.log").write_text("\n".join(en.log) + "\n", encoding="utf-8")
    print(f"写出 {outdir / 'assemblies.json'}（{len(out)} 种拼法）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
