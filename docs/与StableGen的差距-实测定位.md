# 为什么不如 StableGen：两处实测定位（2026-11-04）

## 结论先说

1. **错位重影是真的，已修**：把原始 2D 图直接投影到正面，会因为「网格比例 ≠ 原图比例」
   而整体偏移 —— 画出来的刘海/眉毛正好落在眼睛上，渲染出来就是两张脸叠着。
   StableGen 不投影原图，它用网格渲的深度图**重新生成**正面，所以天然对齐。
   现在四个机位全部按网格重绘（`tools/run_guided.py`）。
2. **但修完还是不如它，第二处原因也量出来了**：Qwen-Image-2.1 在 8 步 Lightning
   下会把网格渲出的彩图**重绘成半写实**，颜色变淡、锐度掉一档。而网格自带的贴图
   本来就清晰、颜色也对。所以「用 AI 重绘」这一步在这张图上其实是**净损失**。

## 一、错位重影：直接投影原图

### 现象

3D 渲染的正脸上出现「两层脸」：不是重影滤镜那种模糊，而是**画上去的眉毛压在渲染
出来的眼睛上**、鼻子有两个、嘴巴错位。

### 为什么

```
原图比例                     网格比例
头/身高 ≈ 1:6.5      ≠     头/身高 ≈ 1:7.3
                ↓
   按包围盒线性映射（头顶对齐、脚底对齐）
                ↓
      中间部位的误差最大 —— 正好是脸
```

`make_aligned_view()` 的做法是「把原图的 alpha 包围盒线性映射到模型的投影包围盒」。
这个映射只有在**两者比例一致**时才正确。TRELLIS 生成的网格是它自己想象的体型，
不可能与原图逐像素同构 —— 所以中间区域必然漂移。

实测（`out/asuka_face_1to1.png`）：直接投影版的两道眉毛落在眼睛上、鼻梁上多一个高光点。
按网格重绘后再投影（`out/asuka_face_fix.png`）重影消失。

### 修法

正面也交给 Qwen 按网格重绘。原图只当「设计参考」，不再参与投影。

```python
# tools/run_guided.py —— 默认 indices 改成 0,1,2,3
# bake 不再传 --reference
cmd = [PY, "tools/bake_projection.py", mesh,
       "--views-dir", ready, "--view-indices", "0,1,2,3",
       "--align", "similarity", "--guide-dir", guide, ...]
```

## 二、真正的差距：Qwen 重绘会「换画风」

`out/asuka_style_test.png` 三联：左 = 网格渲出的彩图，中 = Qwen 重绘（原提示词），
右 = Qwen 重绘 + `VNCCS_QI2_AnimeOverhaulV2` LoRA。

| | 左（渲染） | 中（Qwen） | 右（Qwen+动漫 LoRA） |
|---|---|---|---|
| 画风 | 赛璐璐、平涂 | **偏半写实**，皮肤有渐变 | 略好，仍偏写实 |
| 颜色 | 橙发、正红战服 | **发色变浅金、战服变亮** | 同中 |
| 锐度 | 边缘硬 | 边缘发软 | 同中 |

也就是说：**输入越清晰，这一步丢得越多**。对骑士那类「原生链把脸重绘成金属」的
崩坏案例，重绘是救命；对本身贴图就对的图（这张 Asuka），重绘是净损失。

## 三、StableGen 为什么不会掉档：masked / sequential

读 `D:\code\gen\others\StableGen-main` 的 `stablegen/texturing/generator.py`：

```python
# 第 0 个机位：从输入图直接生成
if self._current_image == 0:
    image = workflow_manager.generate_qwen_edit(...)
# 之后每个机位：
else:
    render_info = self._get_uploaded_image_info(..., "render", camera_id=cur)
    mask_info   = self._get_uploaded_image_info(..., "visibility", camera_id=cur)
    # ↑ 渲当前状态 + **可见性 mask**，然后只重绘「还没被覆盖」的像素
```

配套参数（`ui/presets.py` DEFAULT）：
`generation_method="sequential"`、`sequential_factor=0.7`、`blur_mask=True`、
`denoise=0.8`、`weight_exponent_after_generation=15.0`、
`discard_factor=90`、`controlnet depth strength 0.5`。

**关键就是那条 visibility mask**：已经投好的像素一个都不动，只有空白的区域交给
扩散模型。我们现在的做法是「整幅重绘」，所以会把已经对的部分也改坏。

### 权重公式（`_bake_visibility_weights`，可直接照抄）

```python
w = max(dot(normal, -cam_dir), 0) * density_confidence
# 硬剔除：角度 > discard_factor(80~90°)、被遮挡、贴近轮廓（3px 内深度跳变）
# 边缘羽化：edge = max(|x_c|,|y_c|); low = 1-width; w *= (1-smoothstep(t))^gamma
```

`density_confidence = clamp(projected_span / (0.5*min(res_x,res_y)), 0, 1)`。
正交相机下 projected_span 是常数，所以对我们是恒等 —— 不用抄。

## 四、下一步（按性价比）

1. **按「与参考图不一致」的区域才重绘**（等价于他们的 local_edit，但不需要
   inpaint 支持）：把渲染图与对齐后的参考图做低频色差，差异大的区域才用 Qwen 的
   结果，其余保留渲染。这样骑士的脸会被修，Asuka 的脸原样保留。
2. **sequential + visibility mask**：先用 `mask.png`（已有）渲成「覆盖率图」，
   贴进 GLB 当贴图渲一张 mask 图，Qwen 出图后按它做 composite。
3. **图集分辨率**：Qwen 只能出 1024²，角色占 73% 画幅，头只有 ~90px，
   而图集里头部要 ~600px → 放大 6~8 倍。先把角色**裁满画幅**再喂 Qwen，
   白拿 1.36 倍。
