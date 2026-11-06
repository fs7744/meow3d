"""StableGen v0.5.0 原生节点链的纯逻辑单测（不加载模型）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d import sg_native  # noqa: E402


def _types(p):
    return {n["class_type"] for n in p.values()}


def test_textured_chain_structure():
    p = sg_native.build_prompt(image="x.png")
    t = _types(p)
    for need in ("LoadImage", "UNETLoader", "VAELoader", "CLIPVisionLoader",
                 "InvertMask", "ImageCropToMask", "Trellis2Conditioning",
                 "ModelSamplingSD3", "RescaleCFG", "CFGOverride",
                 "EmptyTrellis2LatentStructure", "KSampler",
                 "VaeDecodeStructureTrellis2", "Trellis2ShapeStage",
                 "Trellis2UpsampleStage", "VaeDecodeShapeTrellis",
                 "RemeshMesh", "DecimateMesh", "Trellis2TextureStage",
                 "VaeDecodeTextureTrellis", "UnwrapMesh",
                 "BakeTextureFromVoxel", "ApplyTextureToMesh", "SaveGLB"):
        assert need in t, need
    assert p[[k for k, v in p.items() if v["class_type"] == "UNETLoader"][0]][
        "inputs"]["unet_name"] == "trellis_2_bf16.safetensors"


def test_defaults_follow_stablegen_v050():
    p = sg_native.build_prompt(image="x.png")
    cond = [v for v in p.values() if v["class_type"] == "Trellis2Conditioning"][0]
    assert cond["inputs"]["image"][0] != "1"      # 走 ImageCropToMask，不是原图
    up = [v for v in p.values() if v["class_type"] == "Trellis2UpsampleStage"][0]
    assert up["inputs"]["target_resolution"] == 1024
    assert up["inputs"]["max_tokens"] == 49152
    ks = [v for v in p.values() if v["class_type"] == "KSampler"]
    assert len(ks) == 4                            # 结构 / 形状 / 高清 / 纹理
    assert {k["inputs"]["sampler_name"] for k in ks} == {"euler"}
    assert [k["inputs"]["scheduler"] for k in ks].count("simple") == 1
    dec = [v for v in p.values() if v["class_type"] == "DecimateMesh"][0]
    assert dec["inputs"]["target_face_count"] == 100000
    bake = [v for v in p.values() if v["class_type"] == "BakeTextureFromVoxel"][0]
    assert bake["inputs"]["texture_size"] == 4096


def test_dynamic_combo_inputs_are_dicts():
    """`sign_mode` / `placement_mode` 是 DynamicCombo：直调节点时必须是 dict，
    给字符串会 `'str' object has no attribute 'get'`（2026-11-06 踩过两次）。"""
    p = sg_native.build_prompt(image="x.png")
    remesh = [v for v in p.values() if v["class_type"] == "RemeshMesh"][0]
    assert isinstance(remesh["inputs"]["sign_mode"], dict)
    assert remesh["inputs"]["sign_mode"]["sign_mode"] == "udf"
    dec = [v for v in p.values() if v["class_type"] == "DecimateMesh"][0]
    assert isinstance(dec["inputs"]["placement_mode"], dict)
    assert dec["inputs"]["placement_mode"]["placement_mode"] == "midpoint"


def test_texture_size_snapped_to_256():
    p = sg_native.build_prompt(image="x.png", texture_size=4000)
    bake = [v for v in p.values() if v["class_type"] == "BakeTextureFromVoxel"][0]
    assert bake["inputs"]["texture_size"] == 3840
    unwrap = [v for v in p.values() if v["class_type"] == "UnwrapMesh"][0]
    assert unwrap["inputs"]["resolution"] == 3840


def test_no_texture_is_shape_only():
    p = sg_native.build_prompt(image="x.png", texture=False)
    t = _types(p)
    assert "SaveGLB" in t
    for gone in ("Trellis2TextureStage", "VaeDecodeTextureTrellis", "UnwrapMesh",
                 "BakeTextureFromVoxel", "ApplyTextureToMesh"):
        assert gone not in t, gone
    # 白模仍走重网格/抽面，SaveGLB 吃的是抽面后的网格
    exp = [v for v in p.values() if v["class_type"] == "SaveGLB"][0]
    src = exp["inputs"]["mesh"][0]
    assert p[src]["class_type"] == "DecimateMesh"


def test_no_remesh_keeps_mesh_node():
    p = sg_native.build_prompt(image="x.png", remesh=False, target_faces=None)
    t = _types(p)
    assert "RemeshMesh" not in t and "DecimateMesh" not in t
    unwrap = [v for v in p.values() if v["class_type"] == "UnwrapMesh"][0]
    src = unwrap["inputs"]["mesh"][0]
    assert p[src]["class_type"] == "VaeDecodeShapeTrellis"


def test_background_removal_branch():
    p = sg_native.build_prompt(image="x.png", remove_background=True)
    t = _types(p)
    assert "LoadBackgroundRemovalModel" in t and "RemoveBackground" in t
    assert "InvertMask" not in t
    crop = [v for v in p.values() if v["class_type"] == "ImageCropToMask"][0]
    assert p[crop["inputs"]["masks"][0]]["class_type"] == "RemoveBackground"


def test_background_color_hex():
    p = sg_native.build_prompt(image="x.png", background="white")
    crop = [v for v in p.values() if v["class_type"] == "ImageCropToMask"][0]
    assert crop["inputs"]["background"] == "#FFFFFF"


def test_topo_order_is_dependency_safe():
    p = sg_native.build_prompt(image="x.png")
    order = sg_native._topo(p)
    assert set(order) == set(p)
    pos = {n: i for i, n in enumerate(order)}
    for nid, node in p.items():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert pos[v[0]] < pos[nid]
