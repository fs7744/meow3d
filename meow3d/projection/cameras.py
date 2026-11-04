"""相机布置：把「方位角等分」升级为「按面法线加权聚类」+ 可控俯仰。

StableGen 默认 8 相机、``normal_weighted``（面法线按面积加权做球面 K-means）
或 ``greedy_coverage``（BVH 验证的可见面覆盖率目标 0.95），并且有 elevation
clamp（``core/properties.py:1048-1062, 1106-1142``、
``cameras/placement.py:59-85``）。meow3d 之前是 ``i*360/N`` 的纯水平环，
头顶、肩窝、腋下、脚底天然覆盖不到。

本模块只做**方向规划**（输出 azimuth/elevation 的 JSON），不碰渲染：
`blender_guidance.py` 与 `blender_bake_views.py` 早就支持 elevation，
`bake.projection_bbox()` 也支持，所以 CLI 把 plan 传下去即可。

坐标系与 `bake.normalized_vertices()` 一致：glTF Y-up 导入 Blender 后是
Z-up，方向 `d = (sin(az)cos(el), -cos(az)cos(el), sin(el))`，
`az=0, el=0` 是正面。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import trimesh

Array = np.ndarray

FRONT = np.array([0.0, -1.0, 0.0])


def normalized_mesh(path) -> tuple[Array, Array]:
    """读 GLB 全部几何，应用 scene graph 变换 → Z-up → 归一化到 [-1,1]。"""
    s = trimesh.load(str(path), force="scene")
    vs, fs, off = [], [], 0
    for node in s.graph.nodes_geometry:
        T, geo = s.graph.get(node)
        g = s.geometry[geo]
        V = trimesh.transform_points(np.asarray(g.vertices, np.float64), T)
        V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)
        F = np.asarray(g.faces, np.int64) + off
        vs.append(V)
        fs.append(F)
        off += len(V)
    if not vs:
        raise ValueError("GLB 里没有几何体")
    V, F = np.vstack(vs), np.vstack(fs)
    lo, hi = V.min(0), V.max(0)
    V = (V - (lo + hi) / 2) * (2.0 / float(np.max(hi - lo)))
    return V, F


def face_frame(path, max_faces: int = 60000,
               exclude_bottom_deg: float = 60.0,
               seed: int = 0) -> tuple[Array, Array]:
    """面法线 + 面积权重。返回 (N,3) 单位法线与 (N,) 权重。

    ``exclude_bottom_deg``：法线朝下超过该角度的面不参与聚类（StableGen 的
    ``exclude_bottom_angle`` 同理），避免把相机浪费在脚底。
    """
    V, F = normalized_mesh(path)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    w = 0.5 * length
    keep = length > 1e-12
    n, w = n[keep], w[keep]
    n = n / length[keep, None]
    if exclude_bottom_deg < 90.0:
        # 法线与 -Z 的夹角 < exclude_bottom_deg → 朝下，剔除
        down = n[:, 2] < -math.sin(math.radians(exclude_bottom_deg))
        n, w = n[~down], w[~down]
    if len(n) > max_faces:
        rng = np.random.default_rng(seed)
        p = w / max(w.sum(), 1e-12)
        pick = rng.choice(len(n), size=max_faces, replace=False, p=p)
        n, w = n[pick], w[pick]
    return n.astype(np.float32), w.astype(np.float32)


def kmeans_on_sphere(n: Array, w: Array, k: int,
                     iters: int = 24, seed: int = 0) -> Array:
    """加权球面 K-means：把法线聚成 k 个方向（相机站位）。"""
    if k <= 0 or len(n) == 0:
        return np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    dirs = [n[rng.integers(len(n))].copy()]
    while len(dirs) < k:
        D = np.asarray(dirs)
        # 不能用 abs(dot)：对跖方向（相机在物体另一侧）同样需要独立机位，
        # 取 abs 会把它们当「已覆盖」，初始化落在两簇之间，收敛出 45° 混合簇。
        sim = (n @ D.T).max(1)
        dirs.append(n[int(np.argmin(sim))].copy())
    D = np.asarray(dirs, np.float32)
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    for _ in range(iters):
        assign = (n @ D.T).argmax(1)
        newD = D.copy()
        for j in range(k):
            sel = assign == j
            if not sel.any():
                continue
            v = (n[sel] * w[sel, None]).sum(0)
            ln = float(np.linalg.norm(v))
            if ln > 1e-8:
                newD[j] = v / ln
        if np.allclose(newD, D, atol=1e-5):
            D = newD
            break
        D = newD
    return D


def direction_to_angles(d: Array, max_elevation: float = 35.0
                        ) -> tuple[float, float]:
    """单位方向 → (azimuth, elevation)，并可 clamp 俯仰。"""
    d = np.asarray(d, np.float64)
    d /= max(float(np.linalg.norm(d)), 1e-9)
    if max_elevation < 90.0:
        z = math.sin(math.radians(max_elevation))
        if abs(d[2]) > z:
            d = d.copy()
            d[2] = math.copysign(z, d[2])
            xy = math.sqrt(max(1e-9, 1.0 - z * z))
            cur = math.hypot(d[0], d[1])
            d[0] *= xy / max(cur, 1e-9)
            d[1] *= xy / max(cur, 1e-9)
    el = math.degrees(math.asin(float(np.clip(d[2], -1, 1))))
    az = math.degrees(math.atan2(d[0], -d[1])) % 360.0
    return round(az, 2), round(el, 2)


def _dedupe(dirs: Sequence[Array], min_deg: float = 22.0) -> List[Array]:
    out: List[Array] = []
    for d in dirs:
        if all(math.degrees(math.acos(float(np.clip(
                np.dot(d / np.linalg.norm(d), o / np.linalg.norm(o)), -1, 1))))
               >= min_deg for o in out):
            out.append(np.asarray(d, np.float32))
    return out


def normal_weighted_plan(mesh_path, count: int = 8,
                         max_elevation: float = 30.0,
                         exclude_bottom_deg: float = 60.0,
                         include_front: bool = True,
                         seed: int = 0) -> List[Dict[str, Any]]:
    """生成 count 个机位；第 0 个固定为正面，其余按法线聚类。

    正面必须保留：提示词的 ``front view``、参考图对齐、人眼检查都依赖它。
    聚类结果若与正面太近会被去重，不足的用等分环补齐。
    """
    n, w = face_frame(mesh_path, exclude_bottom_deg=exclude_bottom_deg, seed=seed)
    k = max(1, count - (1 if include_front else 0))
    dirs = list(kmeans_on_sphere(n, w, k, seed=seed))
    if include_front:
        dirs = [FRONT] + dirs
    dirs = _dedupe(dirs)
    # 不足部分用水平等分环补（保证机位数稳定）
    if len(dirs) < count:
        for i in range(count):
            az = math.radians(360.0 * i / count)
            d = np.array([math.sin(az), -math.cos(az), 0.0], np.float32)
            if len(_dedupe(dirs + [d])) > len(dirs):
                dirs = _dedupe(dirs + [d])
            if len(dirs) >= count:
                break
    dirs = dirs[:count]
    # 正面排第一，其余按 azimuth 排序，方便人眼对照
    head = dirs[0] if include_front else None
    rest = dirs[1:] if include_front else dirs
    rest.sort(key=lambda d: direction_to_angles(d, max_elevation)[0])
    dirs = ([head] if head is not None else []) + rest
    return [{"name": f"v{i:02d}", "index": i,
             **dict(zip(("azimuth", "elevation"),
                        direction_to_angles(d, max_elevation)))}
            for i, d in enumerate(dirs)]


def ring_plan(count: int, indices: Optional[Sequence[int]] = None,
              elevation: float = 0.0) -> List[Dict[str, Any]]:
    """旧的等分水平环，保留作对照 / 兼容。"""
    idx = list(indices) if indices else list(range(count))
    return [{"name": f"v{i:02d}", "index": i,
             "azimuth": 360.0 * i / count, "elevation": elevation}
            for i in idx]


def save_plan(plan: Sequence[Dict[str, Any]], path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(list(plan), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def load_plan(path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def angles_for(plan: Sequence[Dict[str, Any]], index: int
               ) -> tuple[float, float]:
    """按 plan 的 index 取角度；找不到就退回等分环。"""
    for c in plan:
        if int(c.get("index", -1)) == index or c.get("name") == f"v{index:02d}":
            return float(c.get("azimuth", 0)), float(c.get("elevation", 0))
    return 0.0, 0.0
