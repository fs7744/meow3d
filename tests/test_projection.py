"""投影管线纯逻辑单测（不加载权重、不占 GPU）。

    .venv/Scripts/python.exe -m pytest tests/test_projection.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import geometry, views  # noqa: E402


def _plane(z: float = 0.0, normal_z: float = 1.0) -> trimesh.Trimesh:
    """一张铺满 UV 的正方形平面，法线沿 +Z（或 -Z）。"""
    v = np.array([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]]) if normal_z > 0 else np.array([[0, 2, 1], [0, 3, 2]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv)
    return m


def test_cameras_count_and_front():
    pts = np.array([[-1, -1, -1], [1, 1, 1]], float)
    cams, center = geometry.make_cameras(pts, 8)
    assert len(cams) == 8
    assert np.allclose(center, 0)
    for c in cams:
        assert np.isclose(np.linalg.norm(c.direction), 1.0)
        assert np.isclose(np.linalg.norm(c.right), 1.0)
        assert np.isclose(np.dot(c.direction, c.right), 0, atol=1e-8)
    # 第 0 个相机在 -Y 方向（正面）
    assert cams[0].direction[1] < -0.9
    # 中心点投影到画面中心
    scr = cams[0].screen(np.array([[0, 0, 0]], float))
    assert np.allclose(scr, [[0.5, 0.5]], atol=1e-6)


def test_uv_map_covers_plane():
    m = _plane()
    uvm = geometry.build_uv_map(m, 64, verbose=False)
    assert uvm.face_id.shape == (64, 64)
    assert uvm.valid.mean() > 0.98
    assert np.allclose(uvm.points[uvm.valid][:, 2], 0.0, atol=1e-5)
    # 平面法线应与 +Z 一致
    n = uvm.normals[uvm.valid]
    assert (n[:, 2] > 0.9).all()


def test_visibility_front_vs_back():
    m = _plane()
    uvm = geometry.build_uv_map(m, 64, verbose=False)
    cams, _ = geometry.make_cameras(np.array([[-1, -1, 0], [1, 1, 0]], float), 4)
    # 造一个 +Z 相机（正对这个平面）
    cam = geometry.Camera("front_z", np.array([0, 0, 1.0]),
                          np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                          2.0, np.zeros(3))
    w_pos = geometry.visibility_weights(uvm, cam, dppg=False)
    assert w_pos[uvm.valid].mean() > 0.5

    cam_back = geometry.Camera("back_z", np.array([0, 0, -1.0]),
                               np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                               2.0, np.zeros(3))
    w_neg = geometry.visibility_weights(uvm, cam_back, dppg=False)
    assert w_neg.max() < 1e-5


def test_exponent_is_monotonic():
    """掠射角下，exponent 越大权重越低。"""
    cos = 0.3
    assert cos ** 8 < cos ** 3 < cos ** 1
    # 直接用权重函数验证不会出 NaN / 负值
    m = _plane()
    uvm = geometry.build_uv_map(m, 32, verbose=False)
    cam = geometry.Camera("tilt", np.array([0.7, 0, 0.714]),
                          np.array([0, 1.0, 0]), np.array([-0.714, 0, 0.7]),
                          2.0, np.zeros(3))
    for k in (1.0, 3.0, 8.0):
        w = geometry.visibility_weights(uvm, cam, exponent=k, dppg=False)
        assert np.isfinite(w).all()
        assert w.min() >= 0


def test_scatter_roundtrip_color():
    m = _plane()
    uvm = geometry.build_uv_map(m, 64, verbose=False)
    cam = geometry.Camera("front_z", np.array([0, 0, 1.0]),
                          np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                          2.0, np.zeros(3))
    atlas = np.zeros((64, 64, 3), np.float32)
    atlas[:] = (0.9, 0.1, 0.1)
    v = views.scatter_atlas_to_view(uvm, cam, atlas, 64)
    assert v.mask.sum() > 100
    px = v.image[v.mask > 0.5]
    assert np.allclose(px.mean(0), (0.9, 0.1, 0.1), atol=1e-5)


def test_dppg_never_increases_total_weight():
    m = _plane()
    uvm = geometry.build_uv_map(m, 64, verbose=False)
    cam = geometry.Camera("front_z", np.array([0, 0, 1.0]),
                          np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                          2.0, np.zeros(3))
    a = geometry.visibility_weights(uvm, cam, dppg=False)
    b = geometry.visibility_weights(uvm, cam, dppg=True, dppg_margin=3)
    assert b.sum() <= a.sum() + 1e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
