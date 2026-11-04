"""Blender 静默少烘视图的宿主侧护栏（UV layer 上限事故回归测试）。"""
from __future__ import annotations

import numpy as np
import pytest

from meow3d.projection import bake


def _views(tmp_path, names):
    d = tmp_path / "views"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        np.save(d / f"{n}_color.npy", np.zeros((2, 2, 3), np.float32))
        np.save(d / f"{n}_weight.npy", np.zeros((2, 2), np.float32))
    return tmp_path


def test_verify_baked_views_accepts_complete_set(tmp_path):
    _views(tmp_path, ["v00", "v01", "v02"])
    assert bake.verify_baked_views(tmp_path, ["v00", "v01", "v02"]) == \
        ["v00", "v01", "v02"]


def test_verify_baked_views_rejects_missing_view(tmp_path):
    # 复现 8 机位事故：Blender 以 0 退出但只写出 7 个
    _views(tmp_path, [f"v{i:02d}" for i in range(7)])
    with pytest.raises(RuntimeError, match="只烘了 7/8"):
        bake.verify_baked_views(tmp_path, [f"v{i:02d}" for i in range(8)])


def test_verify_baked_views_ignores_stray_png(tmp_path):
    _views(tmp_path, ["v00"])
    (tmp_path / "views" / "v00.png").write_bytes(b"x")
    assert bake.verify_baked_views(tmp_path, ["v00"]) == ["v00"]
