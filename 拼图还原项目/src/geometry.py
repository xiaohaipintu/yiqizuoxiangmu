"""多边形几何工具：面积、边、内角、刚体变换。

坐标一律用图像像素坐标系（y 向下），多边形顶点按逆时针（CCW）排列。
"""
from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def polygon_area(poly: np.ndarray) -> float:
    """鞋带公式，返回正值面积。"""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """图像坐标系 y 向下，因此按数学 y 向上的逆时针等价于 signed_area < 0。"""
    poly = np.asarray(poly, dtype=float)
    if signed_area(poly) > 0:
        poly = poly[::-1].copy()
    return poly


def centroid(poly: np.ndarray) -> np.ndarray:
    """多边形面积形心（不是顶点平均）。"""
    x = poly[:, 0]
    y = poly[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return poly.mean(axis=0)
    cx = ((x + xn) * cross).sum() / (6.0 * a)
    cy = ((y + yn) * cross).sum() / (6.0 * a)
    return np.array([cx, cy])


def perimeter(poly: np.ndarray) -> float:
    d = np.diff(np.vstack([poly, poly[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def edges(poly: np.ndarray) -> List[dict]:
    """返回每条边的几何信息：端点、长度、方向角、中点。"""
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        v = b - a
        length = float(np.hypot(*v))
        ang = math.degrees(math.atan2(v[1], v[0])) % 360.0
        out.append(
            {
                "index": i,
                "a": [float(a[0]), float(a[1])],
                "b": [float(b[0]), float(b[1])],
                "length": length,
                "angle": ang,
                "mid": [float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)],
            }
        )
    return out


def interior_angles(poly: np.ndarray) -> List[float]:
    """每个顶点的内角（度）。"""
    n = len(poly)
    angs = []
    for i in range(n):
        p0 = poly[i - 1]
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        v1 = p0 - p1
        v2 = p2 - p1
        cosv = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        cosv = max(-1.0, min(1.0, cosv))
        angs.append(math.degrees(math.acos(cosv)))
    return angs


def drop_collinear(poly: np.ndarray, tol_deg: float = 8.0, px_dev: float = 0.0) -> np.ndarray:
    """去掉冗余顶点：角度接近 180° 且离邻居连线足够近（< px_dev 像素）才删。

    只按角度删会把"接近直线但其实有拐角"的真顶点吃掉（六边形就是这样丢成五边形的），
    加上偏移距离判据后，真拐角即使 152° 也不会被删。
    """
    poly = np.asarray(poly, dtype=float)
    changed = True
    while changed and len(poly) > 3:
        changed = False
        angs = interior_angles(poly)
        for i in range(len(poly)):
            if abs(180.0 - angs[i]) >= tol_deg:
                continue
            if px_dev > 0.0:
                a = poly[i - 1]
                b = poly[(i + 1) % len(poly)]
                ab = b - a
                L = np.linalg.norm(ab)
                if L < 1e-9:
                    continue
                n = np.array([-ab[1], ab[0]]) / L
                if abs(float((poly[i] - a) @ n)) >= px_dev:
                    continue
            poly = np.delete(poly, i, axis=0)
            changed = True
            break
    return poly


def rotation(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]])


def transform_poly(poly: np.ndarray, R: np.ndarray, t: np.ndarray, mirror: bool = False) -> np.ndarray:
    """先镜像（关于 y 轴），再旋转 R，再平移 t。"""
    p = np.asarray(poly, dtype=float).copy()
    if mirror:
        p[:, 0] *= -1.0
    return (R @ p.T).T + np.asarray(t, dtype=float)


def bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )


# ---------------------------------------------------------------- 快速凸多边形运算
def to_pos(poly: np.ndarray) -> np.ndarray:
    """统一下成正向（signed_area>0，数学坐标系里的逆时针），半平面裁剪靠这个定义内侧。"""
    p = np.asarray(poly, dtype=float)
    return p if signed_area(p) >= 0 else p[::-1].copy()


def clip_halfplane(poly: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """保留直线 a→b 左侧的部分（poly 需为正向凸多边形）。"""
    e = b - a
    out = []
    n = len(poly)
    for j in range(n):
        p = poly[j]
        q = poly[(j + 1) % n]
        sp = e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])
        sq = e[0] * (q[1] - a[1]) - e[1] * (q[0] - a[0])
        if sp >= 0:
            out.append(p)
        if (sp > 0 > sq) or (sp < 0 < sq):
            t = sp / (sp - sq)
            out.append(p + t * (q - p))
    return np.array(out) if out else np.zeros((0, 2))


def convex_overlap_area(pa: np.ndarray, pb: np.ndarray) -> float:
    """两个凸多边形的交叠面积（Sutherland-Hodgman 裁剪），比 shapely 快一个量级。"""
    sub = to_pos(pa)
    clp = to_pos(pb)
    for i in range(len(clp)):
        sub = clip_halfplane(sub, clp[i], clp[(i + 1) % len(clp)])
        if len(sub) == 0:
            return 0.0
    return abs(signed_area(sub))


def hull_area(pts: np.ndarray) -> float:
    """所有碎片顶点构成的外接凸包面积（近似表示拼接轮廓的紧致度）。"""
    hull = cv2.convexHull(np.asarray(pts, dtype=np.float32))
    return abs(float(cv2.contourArea(hull)))
try:
    from shapely.geometry import Polygon as _SPoly
except Exception:
    _SPoly = None


def poly_overlap_area(a, b) -> float:
    """任意（含凹）多边形的交叠面积；shapely 不可用时退回凸多边形近似。"""
    if _SPoly is None:
        return convex_overlap_area(a, b)
    pa = _SPoly(np.asarray(a, float))
    pb = _SPoly(np.asarray(b, float))
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    return float(pa.intersection(pb).area)
