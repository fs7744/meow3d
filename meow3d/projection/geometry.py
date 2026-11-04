"""UV 空间映射 / 相机 / 可见度权重 / DPPG。

按 StableGen Wiki 公开的**思路**独立实现（cos^k 加权、遮挡剔除、深度边界抑制、
Reproject 缓存），未复制其 GPL-3.0 源码；不依赖 nvdiffrast / nvdiffrec。

核心取舍：不在相机图像空间做光栅化，而是**在 UV 图集空间**做一次反向映射——
每个 texel 预先算出它属于哪个三角面、对应哪个 3D 点与法线。这份映射最贵，
缓存成 npz 后，改混合参数只需重算权重（毫秒级），这就是 Reproject。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class UVMap:
    """UV 图集每个 texel 的几何信息。坐标约定：atlas[y, x] ← uv=(x/S, y/S)。"""

    size: int
    face_id: np.ndarray          # (S,S) int32, -1 = 无覆盖
    points: np.ndarray           # (S,S,3) float32, 世界/模型空间位置
    normals: np.ndarray          # (S,S,3) float32, 单位法线
    uv_px: np.ndarray            # (S,S,2) float32, 该 texel 的像素中心 uv（供 remap 用）

    @property
    def valid(self) -> np.ndarray:
        return self.face_id >= 0


def build_uv_map(mesh: Any, size: int = 1024, cache: Optional[Path] = None,
                 verbose: bool = True) -> UVMap:
    """把网格的 UV 三角形光栅化到 (size,size)，逐 texel 记录面/3D 点/法线。

    小三角形（UV 面积 < 0.75 px²）直接写中心 texel，避免几十万次空循环。
    """
    if cache is not None and cache.is_file():
        z = np.load(cache)
        if int(z["size"]) == size and z["face_id"].shape == (size, size):
            if verbose:
                print(f"  UV 映射命中缓存 {cache.name}", flush=True)
            return UVMap(size, z["face_id"], z["points"], z["normals"], z["uv_px"])

    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if uv.shape[0] != verts.shape[0]:
        raise ValueError(f"UV 是逐面还是逐顶点？uv={uv.shape} verts={verts.shape}")
    vnorm = np.asarray(mesh.vertex_normals, dtype=np.float32)

    S = size
    face_id = np.full((S, S), -1, dtype=np.int32)
    pts = np.zeros((S, S, 3), dtype=np.float32)
    nrm = np.zeros((S, S, 3), dtype=np.float32)
    uv_px = np.zeros((S, S, 2), dtype=np.float32)
    gx = (np.arange(S, dtype=np.float32) + 0.5) / S
    uv_px[..., 0] = gx[None, :]
    uv_px[..., 1] = gx[:, None]

    tri_uv = uv[faces]                       # (K,3,2)
    tri_v = verts[faces]                     # (K,3,3)
    tri_n = vnorm[faces]                     # (K,3,3)
    area = 0.5 * np.abs(
        (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1])
        - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1])
    ) * S * S

    order = np.argsort(-area)                # 大面先写，小面后写（小的补缝）
    tiny = 0
    for fi in order:
        t = tri_uv[fi]
        x0 = max(0, int(np.floor(t[:, 0].min() * S)))
        x1 = min(S - 1, int(np.ceil(t[:, 0].max() * S)))
        y0 = max(0, int(np.floor(t[:, 1].min() * S)))
        y1 = min(S - 1, int(np.ceil(t[:, 1].max() * S)))
        if x1 < x0 or y1 < y0:
            continue
        if area[fi] < 0.75:
            cx = int(np.clip(t[:, 0].mean() * S, 0, S - 1))
            cy = int(np.clip(t[:, 1].mean() * S, 0, S - 1))
            face_id[cy, cx] = fi
            pts[cy, cx] = tri_v[fi].mean(0)
            n = tri_n[fi].mean(0)
            nrm[cy, cx] = n / (np.linalg.norm(n) + 1e-8)
            tiny += 1
            continue

        xs = (np.arange(x0, x1 + 1, dtype=np.float64) + 0.5) / S
        ys = (np.arange(y0, y1 + 1, dtype=np.float64) + 0.5) / S
        X, Y = np.meshgrid(xs, ys)
        # 解重心： [u v 1]^T = A @ [w0 w1 w2]^T
        A = np.array([[t[0, 0], t[1, 0], t[2, 0]],
                      [t[0, 1], t[1, 1], t[2, 1]],
                      [1.0, 1.0, 1.0]])
        try:
            inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        flat = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], axis=0)
        w = inv @ flat
        m = (w >= -1e-6).all(axis=0)
        if not m.any():
            continue
        yy, xx = np.unravel_index(np.nonzero(m)[0], X.shape)
        yy = yy + y0
        xx = xx + x0
        wm = w[:, m].T
        wm = np.clip(wm, 0, None)
        wm /= wm.sum(axis=1, keepdims=True) + 1e-8
        sel = face_id[yy, xx] < 0           # 大面优先，不覆盖已有
        if not sel.any():
            continue
        yy, xx, wm = yy[sel], xx[sel], wm[sel]
        face_id[yy, xx] = fi
        pts[yy, xx] = np.einsum("ij,ijk->ik", wm, tri_v[fi][None].repeat(len(yy), 0))
        nn = np.einsum("ij,ijk->ik", wm, tri_n[fi][None].repeat(len(yy), 0))
        nrm[yy, xx] = nn / (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-8)

    out = UVMap(S, face_id, pts, nrm, uv_px)
    if verbose:
        print(f"  UV 映射完成：覆盖 {out.valid.mean():.1%}（{tiny} 个微小面走中心近似）",
              flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, size=S, face_id=face_id, points=pts,
                            normals=nrm, uv_px=uv_px)
    return out


@dataclass
class Camera:
    name: str
    direction: np.ndarray        # 单位向量：从模型中心指向相机
    right: np.ndarray
    up: np.ndarray
    radius: float
    center: np.ndarray           # 模型包围盒中心，投影前要减掉

    def screen(self, points: np.ndarray) -> np.ndarray:
        """(...,3) → (...,2) 归一化屏幕坐标 [0,1]。"""
        q = points - self.center
        sx = np.dot(q, self.right) / (2 * self.radius) + 0.5
        sy = 0.5 - np.dot(q, self.up) / (2 * self.radius)
        return np.stack([sx, sy], axis=-1)

    def depth(self, points: np.ndarray) -> np.ndarray:
        return np.dot(points - self.center, self.direction)


def make_cameras(points: np.ndarray, count: int = 8,
                 elevation_deg: float = 12.0) -> tuple[List[Camera], np.ndarray]:
    """环绕相机：方位角均匀，仰角正负交替；第 0 个固定为正面。"""
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2) * 1.4
    cams: List[Camera] = []
    for i in range(count):
        az = 2 * math.pi * i / count
        el = math.radians(elevation_deg * (1 if i % 2 == 0 else -1))
        d = np.array([math.sin(az) * math.cos(el),
                      -math.cos(az) * math.cos(el),
                      math.sin(el)], dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0])
        r = np.cross(world_up, d)
        r /= np.linalg.norm(r) + 1e-8
        u = np.cross(d, r)
        cams.append(Camera(f"cam{i:02d}", d, r, u, radius, center))
    return cams, center


def visibility_weights(uvm: UVMap, cam: Camera, *, exponent: float = 3.0,
                       discard_deg: float = 90.0, grid: int = 256,
                       dppg: bool = True, dppg_margin: int = 3,
                       dppg_threshold: float = 0.02,
                       dppg_feather: int = 8) -> np.ndarray:
    """单个相机对每个 texel 的投影权重。

    1. cos(法线,视线)^exponent；超过 discard angle 直接 0。
    2. 自遮挡：把 texel 投到 (grid,grid) 屏幕格，每格只保留最近的采样点。
    3. DPPG：深度场梯度找出部件边界，沿边界压低该相机的权重，
       让"看得见干净背景面"的相机去填，避免前景色溢到后面。
    """
    valid = uvm.valid
    w = np.zeros((uvm.size, uvm.size), dtype=np.float32)
    if not valid.any():
        return w

    cos = np.dot(uvm.normals, cam.direction)
    cos = np.clip(cos, -1.0, 1.0)
    cutoff = math.cos(math.radians(discard_deg))
    ok = valid & (cos > cutoff)
    w[ok] = np.power(cos[ok], exponent)

    # --- 自遮挡：屏幕格最小深度 + 软可见度 ---
    # 硬阈值「只保留每格最近的一个 texel」会把 99% 表面判死（格子里的点本就
    # 有厚度差）。改成 exp(-gap/thickness)：正面 gap≈0 保留，被挡住的深面衰减。
    scr = cam.screen(uvm.points)
    dep = cam.depth(uvm.points)
    gx = np.clip((scr[..., 0] * grid).astype(np.int32), 0, grid - 1)
    gy = np.clip((scr[..., 1] * grid).astype(np.int32), 0, grid - 1)
    cell = gy * grid + gx
    # depth = dot(P-center, direction)：方向指向相机，所以**越大越靠近相机**。
    # 早期版本按 min 判可见，等于把背面当正面，贴图会整片错乱。
    max_depth = np.full(grid * grid, -np.inf, dtype=np.float32)
    np.maximum.at(max_depth, cell[ok].ravel(), dep[ok].astype(np.float32))
    gap = max_depth[cell] - dep
    thickness = max(2.0 * cam.radius / grid, cam.radius * 0.02)
    vis = np.exp(-np.clip(gap, 0, None) / thickness)
    w *= np.where(np.isfinite(vis), vis, 0).astype(np.float32)

    # --- DPPG：只在 UV 岛内部、只压最强的少数深度边界 ---
    # UV 图集上「岛与岛之间」本来就有巨大跳变，若把它们当深度边界会全图抑制。
    if dppg and dppg_margin > 0:
        d0 = (dep / max(1e-6, cam.radius)).astype(np.float32)
        inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        gxv = cv2.Sobel(d0, cv2.CV_32F, 1, 0, ksize=3)
        gyv = cv2.Sobel(d0, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gxv ** 2 + gyv ** 2)
        grad[~inner] = 0
        pool = grad[inner]
        if pool.size > 100:
            # 只把最强的一小段梯度当边界（默认约 2%），而不是全图
            cut = float(np.quantile(pool, 1.0 - dppg_threshold))
            edge = ((grad > max(cut, 1e-5)) & inner).astype(np.uint8)
            if edge.any():
                k = max(1, dppg_margin)
                band = cv2.dilate(edge, np.ones((k, k), np.uint8))
                feather = cv2.GaussianBlur(band.astype(np.float32),
                                           (0, 0), max(1.0, dppg_feather / 3))
                w *= (1.0 - 0.6 * np.clip(feather, 0, 1))
    return np.where(valid, w, 0).astype(np.float32)
