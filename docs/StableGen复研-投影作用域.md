# StableGen 复研：之前到底遗漏了什么

> 复研时间 2026-11-04 21:25。起因：用户反馈「StableGen 效果非常棒，是不是遗漏了什么」。
> 结论：**是，而且漏的是最根本的一层——投影发生在哪里。**

## 一句话

**StableGen 的贴图不是「在 UV 图集上画出来的」，而是「渲染时的 3D 表面投影材质」；
UV 贴图是最后 bake 出来的产物。我把投影直接做在了 UV texel 上，所以必碎。**

Wiki *Export* 第一句就是答案：

> "A StableGen texture is a **projection material** that combines the camera images
> **at render time**. To use the result elsewhere, **bake it into ordinary UV textures**,
> then export."

## 证据链（官方 README / Wiki / 源码结构）

| 证据 | 说明 |
|---|---|
| `stablegen/raycast.osl` | OSL shader：输入 `Origin`(相机位置)、`Direction`、`SurfaceNormal`、`CameraFOV`、`AngleThreshold`、`Power`、`RayEpsilon`；对**每个表面点**从相机发射线，判断视锥、遮挡、法线夹角、`Power` 权重。全程在 3D 表面连续域，**与 UV 无关** |
| `stablegen/texturing/projection.py` | 构建 Cycles 材质节点图：`ShaderNodeRaycast`、SVM stack 预算、每相机图像纹理、`_SG_ProjectionBuffer` |
| `stablegen/texturing/rendering.py` | 模块 docstring 直写：「Mask processing, render exports, **visibility, texture baking**, and material helpers」；`projection.py` 从它 import `prepare_baking, bake_texture, unwrap` |
| Wiki *Export* → Bake Textures | 「bakes the projected material … **with Cycles**」；默认 **Texture Resolution 2048**、Render Samples 16、Denoise On、**Unwrap Method = Smart UV Project**，结果写到名为 `BakeUV` 的新 UV 图层 |
| 同页 unwrap 方法表 | **「Cube Project — Fast; suitable for very dense meshes such as TRELLIS.2 output」** —— 官方专门给 TRELLIS.2 稠密网格准备了解法 |
| Wiki *Blending and projection* | 权重是「for every **surface point** and every camera」算的；`Bake transform-stable visibility` 则是「compute visibility once **per vertex**」 |

## 逐条对照：漏了什么

| # | StableGen | 我们的 `meow3d/projection/` | 后果 |
|---|---|---|---|
| **1 根本** | 投影在 **Cycles 材质层 / 3D 表面点**，raycast 判可见性 | 在 **UV 图集 texel** 上，用 `cam.screen(uvm.points)` 采样视图 | xatlas 的 106 个碎片 UV 岛里 texel 与表面点关系跳变 → 满屏碎玻璃 |
| **2 根本** | bake 前**重新 unwrap**（Smart UV / Cube Project）到 `BakeUV`，2048² | 直接复用原生纹理链产出的 xatlas 碎片 UV，且只有 1024² | UV 映射只覆盖 55.9% texel，19 万微小面只写中心像素 |
| **3** | 每相机的 **guidance 图由 Blender 从该相机渲染**（Depth / Normal / Workbench / Viewport），生成视图与相机投影**严格对齐** | 只给 Qwen 文字 prompt，人像构图自由；`cam.screen()` 用固定球半径粗算 | 视图与网格不对齐（Wiki 把 "views no longer line up with the mesh" 明确列为已知失败模式） |
| **4** | **Qwen 2.1 预设要求 RGBA 视图，透明像素排除投影** | 浅灰背景 + 启发式颜色阈值 mask（最低仍 0.15） | 背景色混进贴图 |
| **5** | 相机分辨率按**网格轮廓自适应**（Auto Aspect Ratio / Fit camera resolution） | 固定包围球半径 | 人物在画面中比例不一致 |
| **6** | DPPG / 遮挡用 **raycast 真深度**比较 | UV 图集上 Sobel 梯度近似 | 碎片 UV 岛上必然失效（一度压掉 53.7% texel） |

## 决定性验证（排除「小 bug」的可能）

怀疑过是 `_fill_holes` 用碎片底图打底放大了碎裂，于是加了 `--fill inpaint`：

- `--mode qwen` + 4 张 Qwen 视图 + `fill=base` → 碎玻璃
- 同样配置 `fill=inpaint`（洞改用邻近颜色扩散）→ **仍然碎**，只是黑块少了
- 早先的**单相机对照**（只用 cam0 + Qwen 正面图）也碎

→ 碎裂**不是打底策略、不是多相机加权、不是缺新视角**，就是投影作用域本身。
换 UV、调权重、加视图都救不回来。

## 修正后的架构（仍然不依赖 ComfyUI 服务、不用 NVIDIA 库）

关键：**让 Blender 当投影与烘焙后端**（我们本来就有 Blender 4.5.9），
而不是自己在 UV 空间光栅化。

1. 白模 GLB（`MeshOnly.json` 出，不含 Native 纹理）导入 Blender
2. **重新 unwrap**：稠密网格用 `Cube Project`，或 `Smart UV Project` → 干净的 `BakeUV`
3. 放相机（按轮廓自适应分辨率），**每个相机渲染 depth/normal guidance**
4. 用 guidance + 已贴纹理的投影渲染图作上下文，让 Qwen 生成该视角的 **RGBA** 图
   （这一步可继续用 `tools/qwen_views.py` 的只读硬链接方案，但要改成吃 guidance）
5. 在 Blender 建投影材质：每相机 image texture + raycast 可见性 + `cos^k` 混合
   （对应 `raycast.osl` 的思路，自己实现节点图）
6. **`bpy.ops.object.bake(type='DIFFUSE')`** 到 `BakeUV`，2048²，导出 GLB

全程只用 Blender + Qwen + trimesh，**不调用 nvdiffrast / nvdiffrec**，
也不起 ComfyUI 服务 —— 仍符合「做到项目里面」和商用解禁两个目标。

## 哪些结论不变

- `replace` 模式（正面原图像素直贴、其余保留底图）**仍然有效且最稳**：
  它本质就是 GD-Head / Headshot 3 的「可见区直取」，不依赖连续 UV，
  `out/three_way_compare.png` 里它是三者中最锐利的。
- 商用许可结论不变：投影模式不碰 nvdiffrast/nvdiffrec。
- 6 条单测、Reproject 缓存、DPPG 参数化这些工程件可以留着，
  但要搬进「材质层投影 + bake」的新架构里重新接线。

## 下一步

这是**重写投影后端**，不是调参：需要写 Blender 节点图生成 + bake 脚本，
并把 `qwen_views.py` 改成吃每相机 guidance 图。工作量明显大于本轮的小修。
建议先做一个最小闭环验证：单相机 + 干净 Cube UV + 一次 bake，
确认「bake 出来的贴图不碎」之后再铺开多相机。

## 参考

- <https://github.com/sakalond/StableGen/wiki/Export>（Bake / Unwrap / 投影材质定义）
- <https://github.com/sakalond/StableGen/wiki/Blending-and-projection>
- <https://github.com/sakalond/StableGen/wiki/Cameras>（guidance 渲染、自适应分辨率）
- <https://github.com/sakalond/StableGen/wiki/Architectures-and-presets>（Qwen 2.1 RGBA 视图）
- `stablegen/raycast.osl`、`stablegen/texturing/projection.py`、`rendering.py`（只读结构，未复制 GPL 源码）
- 既有文档：`docs/StableGen评估.md`、`docs/投影贴图管线.md`
