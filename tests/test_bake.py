"""v2 Blender bake 后端的纯逻辑单测（不需要 GPU / Blender）。"""
from __future__ import annotations

import io
import json
import math
import struct

import numpy as np
import pytest
from PIL import Image

from meow3d.projection import bake, glbpatch


def _png(size=(8, 8), color=(200, 30, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _tiny_glb(path, png: bytes) -> dict:
    """手搓一个最小 GLB：两个 bufferView 在贴图**后面**，用来验证偏移平移。"""
    pos = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    idx = struct.pack("<3H", 0, 1, 2)
    png_pad = png + b"\x00" * ((-len(png)) % 4)
    tail = struct.pack("<4f", 1, 2, 3, 4)

    bin_ = pos + idx + b"\x00\x00" + png_pad + tail
    img_off = 36 + 6 + 2
    tail_off = img_off + len(png_pad)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
            {"buffer": 0, "byteOffset": img_off, "byteLength": len(png)},
            {"buffer": 0, "byteOffset": tail_off, "byteLength": 16},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0},
                                    "indices": 1, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 1}}}],
        "textures": [{"source": 0}, {"source": 0}],
        "images": [{"bufferView": 2, "mimeType": "image/png"}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    glbpatch._write_glb(path, gltf, bin_)
    return {"png": png, "img_off": img_off, "tail": tail, "pos": pos, "idx": idx}


def test_glbpatch_swaps_texture_and_keeps_mesh_bytes(tmp_path):
    src = tmp_path / "a.glb"
    meta = _tiny_glb(src, _png(color=(1, 2, 3)))
    new_png = _png(size=(16, 16), color=(9, 8, 7))
    dst = tmp_path / "b.glb"
    info = glbpatch.patch_base_color(src, new_png, dst)

    g, bin_ = glbpatch._read_glb(dst)
    bvs = g["bufferViews"]
    off = bvs[2]["byteOffset"]
    assert bin_[off: off + len(new_png)] == new_png, "贴图字节没换成功"
    assert bvs[2]["byteLength"] == len(new_png)
    # 贴图后面的 bufferView 必须整体平移，且内容不变
    assert bin_[bvs[3]["byteOffset"]: bvs[3]["byteOffset"] + 16] == meta["tail"]
    assert bin_[bvs[0]["byteOffset"]: bvs[0]["byteOffset"] + 36] == meta["pos"]
    assert bin_[bvs[1]["byteOffset"]: bvs[1]["byteOffset"] + 6] == meta["idx"]
    assert g["buffers"][0]["byteLength"] == len(bin_)
    assert info["src_size"] > 0 and info["dst_size"] > 0


def test_glbpatch_keeps_offsets_4byte_aligned(tmp_path):
    """后面 accessor 的 byteOffset 必须仍是 4 的倍数，否则 glTF 校验器会报错。"""
    src = tmp_path / "a.glb"
    _tiny_glb(src, _png(color=(1, 2, 3)))
    dst = tmp_path / "b.glb"
    glbpatch.patch_base_color(src, _png(size=(13, 13)), dst)   # 长度不整齐
    g, _ = glbpatch._read_glb(dst)
    assert g["bufferViews"][3]["byteOffset"] % 4 == 0
    assert g["bufferViews"][0]["byteOffset"] % 4 == 0


def test_glbpatch_rejects_mesh_without_base_color(tmp_path):
    src = tmp_path / "a.glb"
    _tiny_glb(src, _png())
    g, b = glbpatch._read_glb(src)
    g["materials"][0]["pbrMetallicRoughness"].pop("baseColorTexture")
    glbpatch._write_glb(src, g, b)
    with pytest.raises(ValueError, match="baseColorTexture"):
        glbpatch.patch_base_color(src, _png(), tmp_path / "b.glb")


def test_glbpatch_image_size_reads_png_header(tmp_path):
    src = tmp_path / "a.glb"
    _tiny_glb(src, _png(size=(64, 32)))
    assert glbpatch.image_size(src) == (64, 32)


def test_prepare_atlas_keeps_alpha_only_for_blend_or_mask(tmp_path):
    """贴图带 alpha 且材质是 MASK/BLEND 时必须保留原 alpha，否则镂空被填实。"""
    import trimesh
    from trimesh.visual.material import PBRMaterial

    def build(alpha_mode: str):
        m = trimesh.creation.box()
        img = Image.new("RGBA", (16, 16), (10, 20, 30, 0))
        m.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(m.vertices), 2)), image=img,
            material=PBRMaterial(baseColorTexture=img, alphaMode=alpha_mode))
        p = tmp_path / f"{alpha_mode}.glb"
        m.export(p)
        return p

    atlas = tmp_path / "atlas.png"
    Image.new("RGB", (8, 8), (200, 0, 0)).save(atlas)
    masked = bake.prepare_atlas_for_patch(atlas, build("MASK"), 8)
    assert Image.open(masked).mode == "RGBA"
    opaque = bake.prepare_atlas_for_patch(atlas, build("OPAQUE"), 8)
    assert Image.open(opaque).mode == "RGB"


def test_projection_bbox_front_is_upright():
    """glTF Y-up 转 Z-up 后，高度应落在 v 上；正面相机的 right 应是 +X。"""
    # 高 2.0、宽 1.6、厚 0.4 的盒子（已是 Blender Z-up 归一化坐标）
    V = np.array([[x, y, z] for x in (-0.8, 0.8)
                  for y in (-0.2, 0.2) for z in (-1.0, 1.0)], dtype=np.float64)
    u0, v0, u1, v1 = bake.projection_bbox(V, 0.0)
    assert u0 == pytest.approx(0.5 - 0.8 / bake.ORTHO)
    assert u1 == pytest.approx(0.5 + 0.8 / bake.ORTHO)
    assert v0 == pytest.approx(0.5 - 1.0 / bake.ORTHO)
    assert v1 == pytest.approx(0.5 + 1.0 / bake.ORTHO)
    assert (v1 - v0) > (u1 - u0), "高度应占 v 方向，而不是深度"


def test_projection_bbox_back_matches_front_extent():
    """背面相机的投影范围应与正面镜像（宽度相同），不能塌成一条线。"""
    V = np.array([[x, y, z] for x in (-0.8, 0.8)
                  for y in (-0.2, 0.2) for z in (-1.0, 1.0)], dtype=np.float64)
    f = bake.projection_bbox(V, 0.0)
    b = bake.projection_bbox(V, 180.0)
    assert (f[2] - f[0]) == pytest.approx(b[2] - b[0], abs=1e-6)
    assert (f[3] - f[1]) == pytest.approx(b[3] - b[1], abs=1e-6)


def test_aligned_view_uses_alpha_bbox_and_keeps_alpha(tmp_path):
    """透明原图按主体 bbox 映射，透明区必须保留 alpha=0（Blender 侧据此排除权重）。"""
    src = tmp_path / "ref.png"
    im = Image.new("RGBA", (200, 400), (0, 0, 0, 0))
    # 主体只占中间 100x100
    for y in range(150, 250):
        for x in range(50, 150):
            im.putpixel((x, y), (200, 30, 40, 255))
    im.save(src)
    dst = tmp_path / "front.png"
    info = bake.make_aligned_view(src, dst, (0.2, 0.2, 0.8, 0.8), 256)
    assert info["transparent"] > 0.8
    out = np.asarray(Image.open(dst).convert("RGBA"))
    assert out.shape == (256, 256, 4)
    a = out[:, :, 3]
    assert (a > 200).mean() > 0.2, "主体应被映射到目标框"
    assert a[0, 0] == 0 and a[-1, -1] == 0, "框外必须透明"
    # 目标框 y = (1-0.8)*256 .. (1-0.2)*256
    assert (a[51:205, 51:205] > 200).any()


def test_split_side_by_side_picks_target_by_aspect():
    """Qwen 拿到「参考图 + 彩图」两图时，会时不时画成左右两格（左原图右结果）。

    判定要求「两块各占 25% 面积且 x 区间不相交」，再按宽高比挑更接近模型轮廓的
    那个 —— 侧视窄、正面宽，这个判据足够区分。
    """
    mask = np.zeros((200, 400), bool)
    mask[40:180, 20:120] = True      # 左：正面（宽 100）
    mask[40:180, 280:320] = True     # 右：侧视（宽 40）
    keep = bake._split_side_by_side(mask, ref_aspect=40 / 140)
    assert keep is not None
    assert keep[:, :150].sum() == 0, "应丢掉左边的正面"
    assert keep[:, 250:].sum() > 0, "应保留右边的侧视"


def test_split_side_by_side_leaves_connected_figure_alone():
    """角色和分离的披风/武器前后叠着（x 有重叠）时不能切。"""
    mask = np.zeros((200, 200), bool)
    mask[20:180, 60:120] = True      # 身体
    mask[20:90, 100:170] = True      # 后方披风，x 与身体重叠
    assert bake._split_side_by_side(mask, ref_aspect=0.3) is None


def test_split_side_by_side_ignores_small_blobs():
    """单一角色 + 一小块碎屑（<25% 面积）不算并排。"""
    mask = np.zeros((200, 200), bool)
    mask[20:180, 80:120] = True
    mask[190:195, 10:14] = True
    assert bake._split_side_by_side(mask, ref_aspect=0.3) is None


def test_similarity_align_keeps_aspect_and_undoes_drift(tmp_path):
    """Qwen 出图会「画大 + 偏一边」，但宽高比基本是对的。

    对齐必须只做等比缩放 + 平移：分轴拉满会把本来正确的形状捏坏（侧视尤其
    明显 —— bbox 只有身高 1/8 宽，横向拉伸直接变成脏带）。
    """
    # 源图：一个 40×200 的竖条，故意画大并偏左
    src = tmp_path / "gen.png"
    im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    im.paste((200, 60, 60, 255), (40, 60, 80, 260))
    im.save(src)
    dst = tmp_path / "aligned.png"
    # 模型框：高 0.5、宽 0.1（典型侧视），居中
    info = bake.make_similarity_aligned_view(src, dst, (0.45, 0.25, 0.55, 0.75), 400)
    out = np.asarray(Image.open(dst).convert("RGBA"))
    ys, xs = np.where(out[:, :, 3] > 128)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert h == pytest.approx(200, abs=3), "高度应对齐模型框（0.5×400）"
    assert w / h == pytest.approx(40 / 200, abs=0.02), "长宽比必须保持"
    assert abs((xs.min() + xs.max()) / 2 - 200) <= 2, "水平应居中到模型框中心"
    assert abs(ys.max() - int(0.75 * 400)) <= 2, "底边应贴模型框底边"


def test_similarity_align_refines_against_guide_mask(tmp_path):
    """给了指引轮廓时，用 IoU 搜索把尺度和平移都定到轮廓上。

    这里刻意让生成图「画小」，且指引轮廓的形状（更宽更矮）与生成图不同，
    只靠包围盒/居中摆位修不到 —— 必须靠轮廓搜索。
    """
    src = tmp_path / "gen.png"
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    im.paste((255, 255, 255, 255), (60, 40, 80, 160))     # 20×120，偏小
    im.save(src)
    ref = Image.new("L", (400, 400), 0)
    ref.paste(255, (185, 105, 215, 295))                  # 模型轮廓 30×190
    ref_path = tmp_path / "depth.png"
    ref.save(ref_path)
    dst = tmp_path / "aligned.png"
    info = bake.make_similarity_aligned_view(
        src, dst, (0.45, 0.25, 0.55, 0.75), 400, ref_mask=str(ref_path))
    assert info["iou"] > 0.85, "轮廓搜索应把 IoU 拉到 0.85 以上"
    out = np.asarray(Image.open(dst).convert("RGBA"))
    ys, xs = np.where(out[:, :, 3] > 128)
    assert abs((xs.min() + xs.max()) / 2 - 200) <= 6
    assert abs((ys.min() + ys.max()) / 2 - 200) <= 6
    assert 180 <= ys.max() - ys.min() + 1 <= 205, "高度应接近模型轮廓的 190"


def test_frame_aligned_view_does_not_recrop(tmp_path):
    """已同框的视图（引导生成的）必须原样铺满，**不能**再按 bbox 裁剪重排。

    真实踩到的坑：侧视 bbox 只有身高 1/8 宽，整张方图被塞进去会横向压成脏带。
    这里用一个「主体偏左上、只占一小块」的图验证像素位置不被改动。
    """
    src = tmp_path / "side.png"
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for y in range(8, 24):
        for x in range(8, 20):
            im.putpixel((x, y), (10, 200, 30, 255))
    im.save(src)
    dst = tmp_path / "out.png"
    info = bake.make_frame_aligned_view(src, dst, 64)
    assert info["aligned"] is True
    out = np.asarray(Image.open(dst).convert("RGBA"))
    ref = np.asarray(Image.open(src).convert("RGBA"))
    assert np.array_equal(out, ref), "同框视图必须逐像素一致（不缩放不重排）"
    assert info["transparent"] == pytest.approx(1 - 12 * 16 / 64 / 64, abs=1e-3)


def test_frame_aligned_view_resizes_whole_frame(tmp_path):
    """尺寸不同只做整幅缩放，主体在画面里的相对位置不变（不是裁完再放）。"""
    src = tmp_path / "v.png"
    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    im.paste((255, 255, 255, 255), (0, 0, 8, 8))   # 只占左上 1/4
    im.save(src)
    dst = tmp_path / "out.png"
    bake.make_frame_aligned_view(src, dst, 64)
    a = np.asarray(Image.open(dst).convert("RGBA"))[:, :, 3]
    assert a.shape == (64, 64)
    # 32→64 是整幅 2 倍放大：原来 8×8 的白块变成 16×16，仍在左上角
    assert a[4, 4] > 200 and a[12, 12] > 200
    assert a[20, 20] == 0, "缩放没把主体移位"
    assert a[40:, 40:].max() == 0, "右下仍透明"


def test_linear_to_srgb_endpoints_and_monotonic():
    x = np.array([0.0, 0.0031308, 0.1, 0.5, 1.0], np.float32)
    y = bake._linear_to_srgb(x)
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert y[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(y) > 0), "必须单调，否则贴图会反相"
    assert y[2] > x[2], "线性→sRGB 中灰应变亮"


def test_fill_holes_does_not_treat_black_as_hole():
    """回填必须用独立的 filled mask，不能拿颜色是否为 0 判断（黑色贴图会被洗掉）。"""
    S = 32
    acc = np.zeros((S, S, 3), np.float32)
    wsum = np.zeros((S, S), np.float32)
    acc[5:10, 5:10] = 0.0          # 合法纯黑区域
    wsum[5:10, 5:10] = 2.0
    acc[20:25, 20:25] = np.array([1.0, 0.5, 0.25]) * 3.0
    wsum[20:25, 20:25] = 3.0
    out = bake._fill_holes(acc, wsum)
    assert np.allclose(out[5:10, 5:10], 0.0), "黑色区域必须保持黑"
    assert np.allclose(out[20:25, 20:25], [1.0, 0.5, 0.25], atol=1e-5)
    assert np.isfinite(out).all() and (out >= 0).all()


def test_fill_holes_fills_large_region_without_black():
    acc = np.zeros((32, 32, 3), np.float32)
    wsum = np.zeros((32, 32), np.float32)
    acc[:4, :4] = 0.8
    wsum[:4, :4] = 1.0
    out = bake._fill_holes(acc, wsum)
    assert (out.sum(axis=2) > 0).all(), "不能留纯黑洞"
    assert out[30, 30] == pytest.approx(0.8, abs=0.02)


def test_merge_winner_keeps_single_view_color(tmp_path):
    """winner 不能把两个相机的颜色混成中间色（无 raycast 时加权会产生脏色）。"""
    vdir = tmp_path / "views"
    vdir.mkdir()
    S = 8
    c0 = np.zeros((S, S, 3), np.float32) + np.array([1.0, 0.0, 0.0])
    c1 = np.zeros((S, S, 3), np.float32) + np.array([0.0, 0.0, 1.0])
    w0 = np.zeros((S, S), np.float32)
    w1 = np.zeros((S, S), np.float32)
    w0[:, :4], w1[:, :4] = 1.0, 0.2
    w0[:, 4:], w1[:, 4:] = 0.2, 1.0
    np.save(vdir / "a_color.npy", c0)
    np.save(vdir / "a_weight.npy", w0)
    np.save(vdir / "b_color.npy", c1)
    np.save(vdir / "b_weight.npy", w1)
    res = bake.merge(tmp_path, S, blend="winner")
    arr = np.asarray(Image.open(res["atlas"]).convert("RGB"))
    assert arr[0, 0, 0] > 200 and arr[0, 0, 2] < 60, "左半应纯红"
    assert arr[0, 7, 2] > 200 and arr[0, 7, 0] < 60, "右半应纯蓝"
    assert res["blend"] == "winner"


def test_merge_flips_rows_and_writes_srgb(tmp_path):
    """bake 的 pixels 从左下开始；PIL 首行是上，合并输出必须翻行。"""
    vdir = tmp_path / "views"
    vdir.mkdir()
    S = 8
    c = np.zeros((S, S, 3), np.float32)
    c[0] = [1.0, 0.0, 0.0]          # 底部一行 = 红
    c[-1] = [0.0, 0.0, 1.0]         # 顶部一行 = 蓝
    w = np.ones((S, S), np.float32)
    np.save(vdir / "front_color.npy", c)
    np.save(vdir / "front_weight.npy", w)
    res = bake.merge(tmp_path, S)
    arr = np.asarray(Image.open(res["atlas"]).convert("RGB"))
    assert arr[0, 0, 2] > arr[0, 0, 0], "PNG 首行应是 bake 的顶部（蓝）"
    assert arr[-1, 0, 0] > arr[-1, 0, 2], "PNG 末行应是 bake 的底部（红）"
    assert res["coverage"] == pytest.approx(1.0)
