"""`drtk` 的等价替身（StableGen 节点包 nodes/nodes_unwrap.py:1026-1076 用）。

真 `drtk` 是 comfy-env/pixi 渠道包，PyPI 上同名 100.0.0 是占位包（只装了
测试文件，没有 drtk 模块）。这里按节点实际调用的两个 API 给出等价实现：

- `rasterize(verts_uv, tri, height, width)` → `[1,H,W]` 局部三角形 id（-1 背景）
- `render(verts_uv, faces, face_img)` → `(_, bary_img)`，`bary_img[1,3,H,W]`
  为逐像素重心坐标，对应 faces 的 (v0,v1,v2) 顺序

坐标约定（照 nodes_unwrap.py:1046-1050）：`verts_uv = [uv*S-0.5, uv*S-0.5, 1]`，
即 UV 范围 [0,1] 映射到像素坐标 [-0.5, S-0.5]，像素 (i,j) 的中心就是 (i,j)。
UV 空间三角形覆盖用本机已装的 nvdiffrast 光栅化，重心坐标用纯 torch 解算。

⚠️ 许可提示：nvdiffrast 是 NVIDIA Source Code License（仅 research/evaluation）。
本项目原本的 v2 投影后端刻意不用它；这里是为了在进程内复刻 StableGen 的
`Trellis2RasterizePBR` 而引入。商用前需把 `_rasterize_uv` 换成纯 torch 版本
或取得相应授权。
"""
from __future__ import annotations

from typing import Any

import torch

_glctx = None


def _context():
    global _glctx
    if _glctx is None:
        import nvdiffrast.torch as dr
        try:
            _glctx = dr.RasterizeCudaContext()
        except Exception:  # noqa: BLE001
            _glctx = dr.RasterizeGLContext()
    return _glctx


def rasterize(verts_uv, tri, height, width):
    """UV 空间三角形覆盖。

    verts_uv: [1,V,3] float（x,y 为像素坐标，z 恒为 1）
    tri:      [F,3] int，局部面索引
    返回:     [1,H,W] int32，命中为局部 face id，未命中 -1
    """
    import nvdiffrast.torch as dr

    S = int(height)
    W = int(width)
    x, y = verts_uv[0, :, 0], verts_uv[0, :, 1]
    # [-0.5, S-0.5] → clip [-1, 1]，y 翻转（图像行 0 在上方）
    clip_x = (x + 0.5) / W * 2 - 1
    clip_y = 1 - (y + 0.5) / S * 2
    pos = torch.stack([clip_x, clip_y,
                       torch.zeros_like(x), torch.ones_like(x)], dim=-1).unsqueeze(0)
    pos = pos.float().contiguous()
    faces = tri.int().contiguous()
    rast, _ = dr.rasterize(_context(), pos, faces, [S, W])
    # nvdiffrast 的 triangle id 是 1-based，0 表示背景
    ids = rast[0, ..., 3].long() - 1
    # ⚠️ 实测：nvdiffrast 输出数组 y=0 对应 clip y=-1（画面底部），
    # 而节点包把 (0,0) 当左上、y 向下使用（render() 里 py=arange(H)）。
    # 不翻这一下，rast_face_ids 会上下镜像，valid_pos 采样到错误 3D 位置，
    # 贴图变成碎裂迷彩斑。最小三角形验证：左上三角形翻转后 (0,0) 命中。
    ids = ids.flip(-2)
    return ids.unsqueeze(0).int().contiguous()


def render(verts_uv, faces, face_img):
    """逐像素重心坐标。

    face_img: [1,H,W] 全局 face id（-1 背景），由调用方分块合并得到。
    返回 (None, bary_img)，bary_img 形状 [1,3,H,W]。
    """
    H, W = int(face_img.shape[-2]), int(face_img.shape[-1])
    dev = face_img.device
    ids = face_img[0].long()
    valid = ids >= 0
    bary = torch.zeros((1, 3, H, W), dtype=verts_uv.dtype, device=dev)
    if not bool(valid.any()):
        return None, bary

    sel = ids[valid]
    tri = faces.long()[sel]                       # [N,3]
    uv = verts_uv[0, :, :2].float()               # [V,2] 像素坐标
    p0, p1, p2 = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]

    gy = torch.arange(H, device=dev, dtype=torch.float32)
    gx = torch.arange(W, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    px, py = xx[valid], yy[valid]

    denom = ((p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0])
             + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1]))
    safe = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
    w0 = ((p1[:, 1] - p2[:, 1]) * (px - p2[:, 0])
          + (p2[:, 0] - p1[:, 0]) * (py - p2[:, 1])) / safe
    w1 = ((p2[:, 1] - p0[:, 1]) * (px - p2[:, 0])
          + (p0[:, 0] - p2[:, 0]) * (py - p2[:, 1])) / safe
    w2 = 1 - w0 - w1
    # 退化三角形：给一个均匀权重，避免 NaN 传到 3D 位置
    bad = denom.abs() <= 1e-12
    if bool(bad.any()):
        w0 = torch.where(bad, torch.full_like(w0, 1 / 3), w0)
        w1 = torch.where(bad, torch.full_like(w1, 1 / 3), w1)
        w2 = torch.where(bad, torch.full_like(w2, 1 / 3), w2)

    bary[0, 0][valid] = w0.to(bary.dtype)
    bary[0, 1][valid] = w1.to(bary.dtype)
    bary[0, 2][valid] = w2.to(bary.dtype)
    return None, bary


def __getattr__(name: str) -> Any:
    """节点若用到其它 drtk 符号，明确报错而不是静默给 None。"""
    raise AttributeError(f"sg_drtk 未实现 drtk.{name}")
