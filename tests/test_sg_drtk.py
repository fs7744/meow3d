"""drtk 替身的 UV 光栅化方向 / 重心插值回归测试。

背景：nvdiffrast 输出数组 y=0 对应画面底部，而节点包按「(0,0) 左上、
y 向下」使用。少了那次 flip，贴图会变成碎裂迷彩斑（2026-11-05 实测）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d import sg_drtk  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("需要 CUDA（nvdiffrast RasterizeCudaContext）", allow_module_level=True)


def _left_upper_triangle():
    s = 8
    # verts_uv = uv*S-0.5，范围 [-0.5, S-0.5]；这个三角形覆盖左上半
    v = torch.tensor(
        [[[-0.5, -0.5, 1.0], [s - 0.5, -0.5, 1.0], [-0.5, s - 0.5, 1.0]]],
        device="cuda")
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    return s, v, tri


def test_rasterize_uses_top_left_origin():
    s, v, tri = _left_upper_triangle()
    idx = sg_drtk.rasterize(v, tri, s, s)[0]
    assert int(idx[0, 0]) == 0, "左上角应命中"
    assert int(idx[s - 1, s - 1]) == -1, "右下角应为空"
    rows = [int((idx[j] >= 0).sum()) for j in range(s)]
    assert rows == sorted(rows, reverse=True), "命中行数应自上而下递减"


def test_render_barycentric_is_self_consistent():
    s, v, tri = _left_upper_triangle()
    idx = sg_drtk.rasterize(v, tri, s, s)[0]
    # 让 3D 顶点坐标 = 像素坐标/S，则插值出的 valid_pos 应等于像素坐标/S
    verts = torch.stack(
        [v[0, :, 0] / s, v[0, :, 1] / s, torch.zeros(3, device="cuda")], dim=-1)
    _, bary = sg_drtk.render(v, tri, idx.unsqueeze(0))
    mask = idx >= 0
    b = bary[0].permute(1, 2, 0)[mask]
    face_verts = verts[tri.long()[idx[mask].long()]]
    pos = (face_verts * b.unsqueeze(-1)).sum(dim=1)
    gy, gx = torch.meshgrid(torch.arange(s, device="cuda"),
                            torch.arange(s, device="cuda"), indexing="ij")
    expected = torch.stack(
        [gx[mask] / s, gy[mask] / s,
         torch.zeros_like(gx[mask], dtype=torch.float32)], dim=-1)
    assert float((pos - expected).abs().max()) < 1e-5
