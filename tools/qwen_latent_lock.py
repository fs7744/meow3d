"""StableGen `attach_qwen_latent_lock`（graph_validation.py:104）的 diffusers 等价实现。

源码做的是：把 ctx render 用 VAE 编码成 sampler 的起始 latent，再用
`SetLatentNoiseMask(repair_mask)` 让 **白=可去噪、黑=锁定**；ComfyUI 的
noise-mask sampler 每个 step 后把被保护的源 latent 还原回去。

本机 `Engine.txt2img_pipe("qwen-image-2.1")` 返回 `QwenImage21Pipeline`：
* `vae_scale_factor=16`，`num_channels_latents=64`；
* `prepare_latents` 返回主 latent `L=[1, (size/16)^2, 64]` 和条件图 latent
  `IL=[1, N*(size/16)^2, 64]`（N 张条件图沿序列维拼接）。
* 顺序生成时条件图是 `[参考图, ctx render]`，所以 `IL[:, -L.shape[1]:, :]`
  **就是 ctx 的 VAE latent**，与 `L` 形状完全一致。

因此这里不自己碰 VAE（那个 VAE 是 4 通道 RGBA、有 padding，自己编码容易错），
而是 patch `prepare_latents` 复用 pipeline 的编码结果，把起始 latent 换成
`z_ctx*(1-mask) + noise*mask`，并用 `callback_on_step_end` 每步还原锁定区。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class LatentLock:
    def __init__(self, pipe, size: int):
        self.pipe = pipe
        self.size = int(size)
        self.orig_prepare = pipe.prepare_latents
        self.mask: torch.Tensor | None = None
        self.z_ctx: torch.Tensor | None = None
        self.info: dict = {}
        pipe.prepare_latents = self._prepare

    def set_mask(self, mask_p: str | Path, dilate: int = 9) -> None:
        """repair mask 图：**白=可去噪（品红区），黑=锁定**。"""
        import cv2

        im = Image.open(mask_p).convert("L")
        a = np.asarray(im, dtype=np.float32)
        if dilate > 0:
            a = cv2.dilate((a > 128).astype(np.uint8) * 255,
                           np.ones((dilate, dilate), np.uint8)).astype(np.float32)
        # 网格与 pipeline 主 latent 一致：(size / vae_scale_factor)^2
        g = max(1, self.size // int(self.pipe.vae_scale_factor))
        small = cv2.resize(a, (g, g), interpolation=cv2.INTER_AREA) / 255.0
        t = torch.from_numpy(small.reshape(1, g * g, 1).astype(np.float32))
        self.mask = t
        self.info["grid"] = (g, g)
        self.info["repair"] = float(t.mean())

    def _prepare(self, images, batch_size, num_channels_latents, height, width,
                 dtype, device, generator, latents=None):
        L, IL = self.orig_prepare(images, batch_size, num_channels_latents,
                                  height, width, dtype, device, generator,
                                  None)          # 先拿默认噪声 L 与条件 latent IL
        if IL is None or self.mask is None:
            return L, IL
        z = IL[:, -L.shape[1]:, :].to(L.dtype)   # 最后一张条件图 = ctx render
        m = self.mask.to(device=L.device, dtype=L.dtype)
        if m.shape[1] != L.shape[1]:
            g = int(math.sqrt(L.shape[1]))
            m = torch.nn.functional.interpolate(
                m.reshape(1, 1, int(math.sqrt(m.shape[1])),
                          int(math.sqrt(m.shape[1]))),
                size=(g, g), mode="bilinear", align_corners=False
            ).reshape(1, g * g, 1)
        noise = torch.randn(L.shape, generator=generator, device=L.device,
                            dtype=L.dtype)
        self.z_ctx = z
        self.mask = m
        self.info["seq"] = int(L.shape[1])
        return z * (1.0 - m) + noise * m, IL

    def callback(self, _pipe, _step, _t, kwargs):
        lat = kwargs["latents"]
        if self.z_ctx is None or self.mask is None:
            return kwargs
        z = self.z_ctx.to(lat.dtype)
        m = self.mask.to(lat.dtype)
        kwargs["latents"] = z * (1.0 - m) + lat * m
        return kwargs

    def restore(self) -> None:
        self.pipe.prepare_latents = self.orig_prepare
