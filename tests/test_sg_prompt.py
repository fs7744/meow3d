"""StableGen 9 节点 API prompt 的纯逻辑单测（不加载模型）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d import sg_io, sg_prompt  # noqa: E402


def test_prompt_matches_stablegen_defaults():
    p = sg_prompt.build_prompt(image="x.png")
    assert p["2"]["inputs"]["resolution"] == "1024_cascade"
    # 模板写 auto，运行时被 scene.trellis2_precision（默认 bf16）覆盖
    assert p["2"]["inputs"]["precision"] == "bf16"
    assert p["5"]["inputs"]["seed"] == 0
    assert p["5"]["inputs"]["max_tokens"] == 49152
    assert p["5"]["inputs"]["ss_sampling_steps"] == 12
    assert p["5"]["inputs"]["shape_guidance_strength"] == 7.5
    # 模板写 7.5，运行时被 scene.trellis2_tex_guidance=3.0 覆盖
    assert p["6"]["inputs"]["tex_guidance_strength"] == 3.0
    assert p["6"]["inputs"]["tex_sampling_steps"] == 12
    assert p["8"]["inputs"]["target_face_count"] == 100000
    assert p["8"]["inputs"]["remesh"] == "on"
    assert p["8"]["inputs"]["remove_inner_faces"] is False
    assert p["9"]["inputs"]["texture_size"] == 4096
    assert p["9"]["inputs"]["original_mesh"] == ["5", 0]
    assert p["7"]["inputs"]["file_format"] == "glb"


def test_skip_branch_removes_background_node():
    """StableGen mesh_gen/workflows.py:1274-1277。"""
    p = sg_prompt.build_prompt(image="x.png", bg_removal="skip")
    assert "3" not in p
    assert p["4"]["inputs"]["image"] == ["1", 0]
    assert p["4"]["inputs"]["mask"] == ["1", 1]
    q = sg_prompt.build_prompt(image="x.png", bg_removal="auto")
    assert "3" in q
    assert q["4"]["inputs"]["image"] == ["3", 0]


def test_overrides_are_applied():
    p = sg_prompt.build_prompt(image="x.png", seed=7, prefix="abc",
                               texture_size=2048, target_face_count=50000,
                               tex_guidance_strength=5.0)
    assert p["5"]["inputs"]["seed"] == 7 and p["6"]["inputs"]["seed"] == 7
    assert p["9"]["inputs"]["texture_size"] == 2048
    assert p["8"]["inputs"]["target_face_count"] == 50000
    assert p["6"]["inputs"]["tex_guidance_strength"] == 5.0
    assert p["7"]["inputs"]["filename_prefix"].startswith("abc_")


def test_background_color_is_native_node_option():
    """background_color 是 Trellis2GetConditioning 原生选项，不是自创参数。"""
    p = sg_prompt.build_prompt(image="x.png", background_color="white")
    assert p["4"]["inputs"]["background_color"] == "white"
    assert sg_prompt.build_prompt(image="x.png")["4"]["inputs"]["background_color"] == "black"
    with pytest.raises(sg_prompt.PromptError):
        sg_prompt.build_prompt(image="x.png", background_color="pink")


def test_topo_order():
    p = sg_prompt.build_prompt(image="x.png")
    order = sg_prompt._topo(p)
    assert order.index("1") < order.index("4")
    assert order.index("2") < order.index("4")
    assert order.index("5") < order.index("6")
    assert order.index("5") < order.index("8")
    assert order.index("9") < order.index("7")
    assert set(order) == set(p)


def test_expand_dynamic_combo():
    inp = sg_io.Input("remesh", parent=sg_io.DynamicCombo,
                      options=[
                          sg_io.Option("off", [sg_io.Input("fill_holes", default=True)]),
                          sg_io.Option("on", [sg_io.Input("remesh_band", default=1.0),
                                              sg_io.Input("remove_inner_faces", default=False)]),
                      ])
    out = sg_prompt._expand_dynamic(inp, "on",
                                    {"remesh": "on", "remesh_band": 2.5,
                                     "remove_inner_faces": True})
    assert out["remesh"] == "on"
    assert out["remesh_band"] == 2.5          # prompt 平铺值优先
    assert out["remove_inner_faces"] is True
    out2 = sg_prompt._expand_dynamic(inp, "on", {"remesh": "on"})
    assert out2["remesh_band"] == 1.0         # 回落 schema 默认
    assert out2["remove_inner_faces"] is False


def test_load_image_mask_is_foreground(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    p = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 3), (10, 20, 30, 0)).save(p)
    im = Image.open(p).convert("RGBA")
    im.putpixel((0, 0), (200, 100, 50, 255))
    im.save(p)

    image, mask = sg_prompt.load_image(p)
    assert tuple(image.shape) == (1, 3, 4, 3)
    assert tuple(mask.shape) == (1, 3, 4)
    assert float(mask[0, 0, 0]) == 1.0        # 前景=1（与 RemoveBackground 输出一致）
    assert float(mask[0, -1, -1]) == 0.0


def test_shape_only_matches_stablegen_template():
    """`trellis2_texture_mode=qwen_image_21` 时走 shape_only 白模板块。

    照 util/workflow_templates.py:1764-1852 与 mesh_gen/workflows.py:1231-1242、
    1327-1330。bg_removal=skip 会再删掉 3 号 RemoveBackground。
    """
    p = sg_prompt.build_prompt(image="x.png", skip_texture=True, seed=9,
                               target_face_count=50000)
    assert {n["class_type"] for n in p.values()} == {
        "LoadImage", "LoadTrellis2Models", "Trellis2GetConditioning",
        "Trellis2ImageToShape", "Trellis2Simplify", "Trellis2ExportTrimesh",
    }
    assert "3" not in p                        # skip 分支
    assert p["4"]["inputs"]["image"] == ["1", 0]
    assert p["4"]["inputs"]["mask"] == ["1", 1]
    assert p["2"]["inputs"]["precision"] == "bf16"
    assert p["5"]["inputs"]["seed"] == 9
    assert p["6"]["class_type"] == "Trellis2Simplify"
    assert p["6"]["inputs"]["trimesh"] == ["5", 0]
    assert p["6"]["inputs"]["target_face_count"] == 50000
    assert p["6"]["inputs"]["remesh"] is True  # qdc 服务端 remesh
    assert p["7"]["inputs"]["trimesh"] == ["6", 0]
    # 白模板块里没有 texture 相关节点
    assert not any(n["class_type"] in ("Trellis2ShapeToTexturedMesh",
                                       "Trellis2RasterizePBR")
                   for n in p.values())


def test_shape_only_topo_and_default_unchanged():
    p = sg_prompt.build_prompt(image="x.png", skip_texture=True)
    order = sg_prompt._topo(p)
    assert order.index("5") < order.index("6") < order.index("7")
    assert set(order) == set(p)
    # 默认仍是 native 9 节点链，不受影响
    q = sg_prompt.build_prompt(image="x.png")
    assert len(q) == 8                        # 9 节点模板删掉 remove_bg
    assert q["6"]["class_type"] == "Trellis2ShapeToTexturedMesh"
    assert q["9"]["class_type"] == "Trellis2RasterizePBR"


def test_shape_only_keeps_native_options():
    p = sg_prompt.build_prompt(image="x.png", skip_texture=True,
                               bg_removal="auto", background_color="gray")
    assert "3" in p
    assert p["4"]["inputs"]["image"] == ["3", 0]
    assert p["4"]["inputs"]["background_color"] == "gray"
