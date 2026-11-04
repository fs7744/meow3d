"""裁满画幅（--fill-frame）的纯逻辑单测。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from meow3d.projection import bake


def _figure():
    """细长人形包围盒：高 2.0、水平只有 0.6（与归一化后的人物一致）。"""
    return np.array([[x, y, z]
                     for x in (-0.3, 0.3)
                     for y in (-0.2, 0.2)
                     for z in (-1.0, 1.0)], np.float64)


ANGLES = [(0, 0), (54, -1), (107, -2), (135, 0),
          (171, 19), (204, -11), (277, 0), (359, 30)]


def test_fit_ortho_tightens_frame():
    o = bake.fit_ortho(_figure(), ANGLES, margin=1.08)
    assert 2.0 < o < 2.4, o
    assert o < bake.ORTHO
    # 角色占画幅比例：2.0/2.8=71.4% → 2.0/o≈92.6%
    assert 2.0 / o > 0.9


def test_fit_ortho_never_clips_any_camera():
    V = _figure()
    o = bake.fit_ortho(V, ANGLES, margin=1.08)
    for az, el in ANGLES:
        u0, v0, u1, v1 = bake.projection_bbox(V, az, el, ortho=o)
        assert -1e-6 <= u0 < u1 <= 1.0 + 1e-6, (az, el, u0, u1)
        assert -1e-6 <= v0 < v1 <= 1.0 + 1e-6, (az, el, v0, v1)


def test_projection_bbox_respects_ortho_override():
    V = _figure()
    wide = bake.projection_bbox(V, 0, 0)                    # 2.8
    tight = bake.projection_bbox(V, 0, 0, ortho=2.16)
    assert (tight[3] - tight[1]) > (wide[3] - wide[1]) * 1.2
    # 中心不变，只是放大
    assert abs((wide[0] + wide[2]) / 2 - (tight[0] + tight[2]) / 2) < 1e-6


def test_fit_ortho_margin_is_enforced():
    a = bake.fit_ortho(_figure(), [(0, 0)], margin=1.01)
    b = bake.fit_ortho(_figure(), [(0, 0)], margin=1.20)
    assert a >= 2.02
    assert b > a


def test_fit_ortho_empty_angles_falls_back():
    assert bake.fit_ortho(_figure(), [], margin=1.08) == bake.ORTHO


def test_reproject_rejects_mismatched_ortho(tmp_path, monkeypatch):
    """画幅变了不能复用旧 npy，否则视图与 ProjUV 错位。"""
    monkeypatch.setattr(bake, "blender_exe", lambda: tmp_path / "fake.exe")
    work = tmp_path / "bake"
    (work / "views").mkdir(parents=True)
    (work / "bake_config.json").write_text(
        json.dumps({"size": 1024, "ortho": 2.8}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ortho"):
        bake.run("x.glb", tmp_path / "out.glb", [], size=1024,
                 ortho=2.16, work_dir=work, reproject=True, verbose=False)
