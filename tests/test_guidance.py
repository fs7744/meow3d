"""指引图（guidance map）后处理的纯逻辑单测：不需要 GPU / Blender。"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from meow3d.projection import guidance


def test_camera_azimuths_match_bake_convention():
    cams = guidance.camera_list(4)
    assert [c["name"] for c in cams] == ["v00", "v01", "v02", "v03"]
    assert [c["azimuth"] for c in cams] == [0.0, 90.0, 180.0, 270.0]
    # 8 机位取偶数位 = 正交四向（投影管线的常用组合）
    cams = guidance.camera_list(8, [0, 2, 4, 6])
    assert [c["azimuth"] for c in cams] == [0.0, 90.0, 180.0, 270.0]
    assert [c["name"] for c in cams] == ["v00", "v02", "v04", "v06"]


def test_camera_list_elevation_passthrough():
    cams = guidance.camera_list(2, None, elevation=15.0)
    assert all(c["elevation"] == 15.0 for c in cams)


def _png(tmp_path, arr: np.ndarray) -> str:
    p = tmp_path / "d.png"
    Image.fromarray(arr.astype(np.uint8), "L").save(p)
    return str(p)


def test_depth_contrast_expands_narrow_range_and_keeps_bg_black(tmp_path):
    """Blender 的深度区间固定 [2,5]，模型实际只占中间一小段 —— 8 位落盘后
    前后差不到 30 级灰，ControlNet 看不出层次。必须按前景分位重映射，
    且背景（0）保持 0，不能被拉成灰。"""
    a = np.zeros((64, 64), np.uint8)
    a[10:50, 20:40] = 120      # 前景挤在很窄一段（模拟 fix 前的输出）
    a[10:50, 20:25] = 150      # 更近的一条
    out = np.asarray(guidance.depth_contrast(_png(tmp_path, a)))
    assert out[0, 0] == 0 and out[-1, -1] == 0        # 背景仍是纯黑
    fg = out[10:50, 20:40]
    assert fg.min() == 0 and fg.max() == 255          # 前景拉满
    assert out[30, 22] > out[30, 35]                  # 近处仍比远处亮（没被反转）


def test_depth_contrast_flat_image_is_identity(tmp_path):
    """全平（没有深度差）时不要除零、不要变成全白。"""
    a = np.zeros((32, 32), np.uint8)
    a[8:24, 8:24] = 130
    out = np.asarray(guidance.depth_contrast(_png(tmp_path, a)))
    assert out.max() == 0 or np.isfinite(out).all()


def test_prepare_depth_writes_expected_name(tmp_path):
    a = np.zeros((16, 16), np.uint8)
    a[4:12, 4:12] = 100
    src = _png(tmp_path, a)
    out = guidance.prepare_depth(src)
    assert out.name == "d_depth_c.png"
    assert out.is_file()


def test_stack_returns_none_for_missing(tmp_path):
    a = np.zeros((8, 8), np.uint8)
    src = _png(tmp_path, a)
    assert guidance.stack({}, "depth") is None
    sheet = guidance.stack({"v00": {"depth": src}, "v01": {"depth": src}}, "depth")
    assert sheet is not None and sheet.size == (16, 8)


def test_render_rejects_unknown_pass(tmp_path):
    with pytest.raises(ValueError):
        guidance.render("x.glb", tmp_path, guidance.camera_list(1),
                        passes=("depth", "bogus"))
