# StableGen 评估（针对 meow3d）

> 对象：<https://github.com/sakalond/StableGen>
> 核实时间 2026-11-04；版本 0.4.0（2026-10-28），仓库当天仍有更新，1452 stars / 141 forks，GPL-3.0。
> 结论基于官方 README、Wiki（Blending and projection / Generation modes /
> TRELLIS.2 mesh generation）、installer.py 与 GitHub API，不靠猜。

## 一句话判断

**它正是我们上一轮调研里「路线 D：多视角投影贴图」的产品级开源实现**，
比我们之前自己写的 `2d-game-meow/workflows/texproject.py`（单视角 + 三角形光栅化 +
cv2.inpaint）完整一个数量级。但它**不能直接当最终形态**（Blender 插件 + ComfyUI
服务器、GPL-3.0 传染），而且**它暴露了一个我们之前漏掉的商用许可炸弹**。

---

## 一、它现在是什么（注意：和旧版不一样了）

0.4.0 已不只是「给已有模型贴图的 Blender 插件」，而是一条龙：

1. **TRELLIS.2 生成网格**（从 prompt 或图片），经
   **`PozzettiAndrea/ComfyUI-TRELLIS2`**（MIT，578 stars），
   检测标志是 `Trellis2ImageToShape` 节点 —— **不是我们用的 visualbruno 那套**。
2. 可选 **SkinTokens** 自动绑骨，支持 FBX/BVH（Mixamo）动作重定向。
3. **环绕相机生成视图 → 投影回网格**（核心，见下节）。
4. **Marigold/StableDelight** 估 albedo/roughness/metallic/normal，烘焙成标准贴图。
5. 导出 glTF（Godot/web）或 FBX（Unreal/Unity），带绑定和动画。

环境：Blender 4.2–4.5 或 5.1+（**5.0 不支持**）；必须有 ComfyUI 服务
（本机或局域网）；SDXL 约 8GB 显存、Qwen Image 需 16GB+；10–50GB 模型磁盘。

### 纹理模式（关键分叉）

| 模式 | 做法 | 用不用 NVIDIA 非商用库 |
|---|---|---|
| **None** | 只出形状 | 不用 |
| **Native (TRELLIS.2)** | TRELLIS.2 自己生成 PBR 并烘焙 UV（texture size 默认 4096） | **用 nvdiffrast / nvdiffrec** |
| **SDXL / Qwen Image Edit / Qwen Image 2.1 / FIBO 等** | 网格不带贴图 → 环绕相机生成视图 → 投影混合 | **完全不用** |

> 我们 meow3d 现在跑的就是「Native」这一类（visualbruno 的原生纹理链）。

---

## 二、它的投影算法（最值得借鉴的部分）

从 Wiki 的 *Blending and projection* 与 *Generation modes* 逐条摘出：

- **逐相机生成 + 可见度加权混合**：每个表面点对每个相机算权重
  `cos(angle)^exponent`（exponent 默认 3，Voronoi 预设到 1000）；
  相机框外 / 被遮挡 / 背向超过 Discard angle（默认 90°，建议 65°）→ 权重 0；
  没被任何相机看到的点用 Fallback Color。
- **DPPG（Depth/Part Projection Guard）**：在深度与物体/材质边界处，
  **抑制每个相机沿边界的一条投影带**（默认 3px、可 2× 超采样、12px 羽化），
  让"看得到干净背景面"的相机去填它。
  → 这正好治我们单视角投影里「手在裙子前面 → 前景色溢到后面」的老问题，
  也解释了为什么我们之前的手抓脚姿势会糊。
- **Sequential context**：Generate Sequentially 一次生成一个相机，
  把**当前已混合纹理的渲染图**作为 context 送入，只 inpaint 尚未覆盖的区域；
  `Protect reliable context` 保护早期可靠视图（最低可信质量默认 0.7）。
- **Guidance Map**：SDXL 走 ControlNet depth（installer 里是
  `xinsir/controlnet-depth-sdxl-1.0`）+ IPAdapter；
  Qwen 系列**不用 ControlNet**，改用 Depth / Normal / Workbench / Viewport
  渲染图当参考图 + 指令 prompt。
  → 我们之前判断「Qwen 无深度 ControlNet，多视角会把侧背掰成正面」，
  StableGen 的答案是**用渲染的深度/法线图当结构参考**，不是等 ControlNet。
- **Reproject**：改混合参数后从已保存的相机图重算投影，**不必重新生成**。
- **Masked Edit / Local Edit / UV Inpaint Missing Areas**：涂哪改哪；
  也能在 UV 空间补未覆盖区域（我们之前用 cv2.inpaint 做同样的事）。
- 相机：默认 8 个（2–32），Normal-Weighted 放置、Greedy Coverage 目标 0.95。

**对照我们自写的 texproject.py**：单视角、三角形光栅化、固定底边对齐、
cv2.inpaint 补洞、UV 利用率低、侧背靠猜 —— StableGen 每一项都有对应成熟解法。

---

## 三、⚠️ 最重要的发现：我们当前链路有商用许可风险

StableGen Wiki 的 *Third-party licenses* 明确列出：

| 组件 | 许可 | 商用 |
|---|---|---|
| TRELLIS.2 / TRELLIS.2-4B 权重 / ComfyUI-TRELLIS2 / BiRefNet / FlexGEMM / CuMesh / O-Voxel | MIT（DINOv3 自有许可） | 可 |
| **nvdiffrast / nvdiffrec (NVIDIA)** | **NVIDIA Source Code License** | **不可，仅 research/evaluation，禁止任何直接或间接货币收益；只有 NVIDIA 及关联方可商用** |

而且 **nvdiffrast / nvdiffrec 只在 Native (TRELLIS.2) 纹理模式**下用于
UV 光栅化与 PBR 烘焙；**projection-based 模式（SDXL / Qwen Image Edit 等）完全不用**。

我们 meow3d 恰好：
- 装了 `nvdiffrast-0.4.0` 与 `nvdiffrec_render` 两个 cp312 轮子；
- 跑的是 visualbruno 的**原生纹理链**（`MeshTexturing` → xatlas UV → 光栅化烘焙）。

也就是说：**商用前不止要换掉 Qwen RESEARCH LICENSE 底模，
还得解决 NVIDIA 这两个库**。此前项目的许可记录只写了 Qwen，漏了这一条。

**StableGen 顺手给出了绕法**：把纹理层从 Native 换成投影模式
（SDXL 或 Qwen Image Edit），整个管线就只剩 MIT/宽松许可组件。
注意 Qwen Image 2.1 / Turbo、Bria FIBO 是非商用，但 **Qwen Image Edit 和 SDXL 不受该限制**
（Wiki 原文；具体模型许可仍建议自行复核）。

> 这条和上一轮「为了脸建议走投影/贴图替换」**指向同一个方向**，
> 只是理由从画质变成了商业可行性 —— 于是这条改造值得优先做。

---

## 四、它能不能治"脸丑"

**部分能，但不根治。**

- 能：可以把**原图当风格/参考图**（`Use External Image as Style`、
  `Reference source` 可选 TRELLIS.2 input image）；用 **Masked Edit** 只重绘脸部；
  用 Depth/Normal 参考图让视图贴合几何；Sequential + Protect reliable context
  减少身份漂移；Local Edit 有颜色匹配（MKL/Histogram/Reinhard…）避免接缝色差。
- 不能：它**没有人脸专用模块**，TRELLIS.2 生成的头部几何本身还是糊
  （32³ 体素 → shape SLat），投影贴图只换"画"，不会长出鼻梁眼窝。
  视图仍是**扩散重绘**而非原图像素直取，写实五官仍有漂移风险。

对我们的两类素材：
- **赛璐璐骑士**（大色块硬线条）：重绘很合适，StableGen 这条路大概率明显改善；
- **写实汉服**（细密五官）：仍要做「原图脸按 UV 直贴」，或走上一轮调研里的
  GD-Head / Headshot 3 式"可见区直取 + 遮挡区补全"。

---

## 五、为什么不能直接搬进来

| 障碍 | 说明 |
|---|---|
| **架构冲突** | Blender 插件 + 独立 ComfyUI 服务。用户已明确「不是用 comfyui 搭，是要做到项目里面」。meow3d 现在是纯进程内 Python，不启服务 |
| **GPL-3.0 传染性** | 复制它的代码进 meow3d，整个项目须 GPL。**只能读文档、按思路自己实现**，不能 copy 源码 |
| **依赖重** | 要装 Blender 扩展、ComfyUI、SDXL/Qwen 全套（10–50GB）、Marigold、SkinTokens；Blender 版本还被卡（5.0 不支持） |
| **换了节点包** | 它用 `PozzettiAndrea/ComfyUI-TRELLIS2`，节点名 `Trellis2ImageToShape`，与我们 visualbruno 的 80 个节点不通用 |

---

## 六、建议（按优先级）

1. **先记许可账**：把 nvdiffrast/nvdiffrec 的非商用限制写进项目文档与发版检查表；
   商用目标是「几何仍用 TRELLIS.2（MIT），纹理改走投影」。
2. **移植算法思路，不移植代码**：在 meow3d 里自己实现
   多相机 + `cos(angle)^k` 可见度加权 + DPPG 边界抑制 + 正视图上下文 +
   UV 补洞 + Reproject 缓存。我们已有 Blender 渲染、trimesh UV、抠图与
   Qwen 编辑能力，缺的只是这套投影/混合逻辑。
3. **拿它当原型对照**：临时装一次 StableGen，用同一张 CC0 骑士图跑
   「Native」vs「SDXL/Qwen 投影」，确认贴图质量上限再决定移植范围。
   这一步是验证，不是最终架构。
4. **脸部单独处理**：Masked Edit 式局部重绘 + 原图脸 UV 直贴，
   两条都留；赛璐璐用前者，写实脸用后者。

---

## 参考（官方文档 / 开源仓库）

- StableGen README — <https://github.com/sakalond/StableGen>
- Wiki: Blending and projection / Generation modes / TRELLIS.2 mesh generation
- installer.py（依赖与模型 URL）
- <https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2> (MIT)
- <https://github.com/visualbruno/ComfyUI-Trellis2>（本项目当前依赖）
- 上一份调研：`meow3d/docs/单图3D人脸质量-业界方案调研.md`
