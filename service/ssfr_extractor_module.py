"""
SSFR-R (Spectral-Spatial Forensic Residual) Feature Extractor Module
替代 LaRE，无需 SDXL UNet (7GB)，使用 MicroForensicUNet (~0.9M) + 信号处理

输出 [B, 7, 32, 32]:
  Ch0: UNet去噪残差 (学习型逆向，GPU batch)
  Ch1: JPEG重压缩残差 (CPU ThreadPool 并行)
  Ch2: 相位一致性 (GPU F.unfold + batch FFT)
  Ch3: 噪声模型偏差 (GPU F.unfold + batch correlation)
  Ch4: SRM跨通道一致性 (GPU Conv2d + F.unfold + batch correlation)
  Ch5: 频谱偏差 (GPU F.unfold + batch FFT + 向量化 radial profile + batched lstsq)
  Ch6: VAE重建误差 (Flux VAE encode→decode, GPU batch)
       — Flux 生成图重建误差极低 (分布内); 真实图/SDXL 误差高 (分布外)

  Ch2-Ch5 消灭了所有 Python patch 循环，全量 GPU 向量化。
"""

import io
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


# ══════════════════════════════════════════════════════════════════════════════
# MicroForensicUNet v2 — RGB 3ch 输入, base_ch=48 (~3.6M 参数)
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MicroForensicUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=48):
        super().__init__()
        self.in_ch = in_ch
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_forensic_unet(real_image_paths, save_path, epochs=40, batch_size=16,
                        lr=1e-3, device="cuda"):
    model = MicroForensicUNet(in_ch=3, base_ch=48).to(device)
    n_params = model.count_params()
    print(f"  MicroForensicUNet v2 (RGB, base48) 参数量: {n_params:,} ({n_params * 4 / 1024 / 1024:.1f} MB)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"  加载 {len(real_image_paths)} 张真实图像 (RGB)...")
    images = []
    for p in tqdm(real_image_paths, desc="  加载真实图"):
        try:
            img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
            images.append(np.array(img, dtype=np.float32) / 255.0)  # [256,256,3]
        except Exception:
            continue
    if len(images) < 50:
        print(f"  ⚠ 真实图像太少 ({len(images)}), 跳过训练")
        return None
    X = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)  # [N, 3, 256, 256]
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         drop_last=True, num_workers=0)
    # 3ch Laplacian kernel: 对每个通道独立做边缘检测
    lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                         dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    lap_k = lap_k.repeat(3, 1, 1, 1)  # [3, 1, 3, 3] for groups=3
    print(f"  开始 RGB 去噪训练 ({epochs} epochs, {len(images)} 张真实图)...")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)  # [B, 3, 256, 256]
            sigma = (torch.rand(batch.size(0), 1, 1, 1, device=device) * 40 + 10) / 255.0
            noise = torch.randn_like(batch) * sigma
            noisy = torch.clamp(batch + noise, 0, 1)
            # 电商鲁棒性增强: 随机 JPEG 压缩模拟 (30% 概率)
            if np.random.rand() < 0.3:
                noisy_np = (noisy.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # [B,H,W,3]
                for bi in range(noisy_np.shape[0]):
                    q = np.random.randint(30, 96)
                    _, buf = cv2.imencode('.jpg', cv2.cvtColor(noisy_np[bi], cv2.COLOR_RGB2BGR),
                                          [cv2.IMWRITE_JPEG_QUALITY, q])
                    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    noisy_np[bi] = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                noisy = torch.from_numpy(noisy_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
            noise_pred = model(noisy)
            mse_loss = F.mse_loss(noise_pred, noise)
            # 逐通道边缘损失 (groups=3)
            edge_loss = F.mse_loss(
                F.conv2d(noise_pred, lap_k, padding=1, groups=3),
                F.conv2d(noise, lap_k, padding=1, groups=3))
            # 跨通道一致性损失: 鼓励模型捕获 R/G/B 噪声的相关结构
            cc_loss = F.mse_loss(noise_pred[:, 0] - noise_pred[:, 1],
                                 noise[:, 0] - noise[:, 1]) + \
                      F.mse_loss(noise_pred[:, 1] - noise_pred[:, 2],
                                 noise[:, 1] - noise[:, 2])
            loss = mse_loss + 0.3 * edge_loss + 0.1 * cc_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:>2}/{epochs}  loss={np.mean(losses):.6f}")
    print(f"  训练完成: {time.time() - t0:.1f}s")
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  模型已保存: {save_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SSFR Extractor — 全 GPU 向量化，消灭 Python patch 循环
# ══════════════════════════════════════════════════════════════════════════════

class SsfrExtractor:
    def __init__(self, device="cuda", unet_path=None, vae_model_id=None):
        self.device = device
        self.out_size = 32
        self.patch_size = 64
        self.stride = 32
        self._n_hw = (512 - self.patch_size) // self.stride + 1  # 15

        # UNet (v2: RGB 3ch, base_ch=48)
        self.unet = None
        if unet_path and Path(unet_path).exists():
            ckpt = torch.load(unet_path, map_location=device, weights_only=True)
            # 自动检测旧模型 (1ch, base32) vs 新模型 (3ch, base48)
            first_key = next(iter(ckpt))
            old_in_ch = ckpt[first_key].shape[1]  # enc1.block.0.weight: [out, in, k, k]
            old_base = ckpt[first_key].shape[0]
            self.unet = MicroForensicUNet(in_ch=old_in_ch, base_ch=old_base).to(device)
            self.unet.load_state_dict(ckpt)
            self.unet.eval()
            print(f"[SSFR] UNet loaded from {unet_path} (in_ch={old_in_ch}, base={old_base})")
        else:
            print(f"[SSFR] ⚠ UNet not found at {unet_path}, Ch0 will be zeros")

        # VAE (Ch6: Reconstruction Error — 检测 Flux 等强生成模型)
        self.vae = None
        if vae_model_id:
            try:
                from diffusers import AutoencoderKL
                print(f"[SSFR] Loading VAE from {vae_model_id} ...")
                # 先尝试作为独立 VAE 仓库加载，失败后尝试 subfolder="vae"
                try:
                    self.vae = AutoencoderKL.from_pretrained(
                        vae_model_id, torch_dtype=torch.float16
                    ).to(device).eval()
                except (OSError, ValueError):
                    self.vae = AutoencoderKL.from_pretrained(
                        vae_model_id, subfolder="vae", torch_dtype=torch.float16
                    ).to(device).eval()
                for p in self.vae.parameters():
                    p.requires_grad_(False)
                print(f"[SSFR] VAE loaded ({sum(p.numel() for p in self.vae.parameters())/1e6:.1f}M params)")
            except Exception as e:
                print(f"[SSFR] ⚠ VAE load failed: {e}, Ch6 will be zeros")
                self.vae = None
        else:
            print(f"[SSFR] VAE not configured, Ch6 will be zeros")

        # SRM kernels (GPU fixed)
        q1 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], np.float32) / 4.
        q2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], np.float32) / 4.
        q3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32) / 4.
        self.srm_kernel_t = torch.from_numpy(np.stack([q1,q2,q3])).unsqueeze(1).to(device)  # [3,1,5,5]

        # Radial index map for Ch5 (precompute)
        ps = self.patch_size
        yy, xx = torch.meshgrid(torch.arange(ps), torch.arange(ps), indexing="ij")
        r_map = torch.round(torch.sqrt((yy - ps//2).float()**2 + (xx - ps//2).float()**2)).long()
        self.inscribed_r = ps // 2   # 32  — limit for valid circular region
        self.max_r = r_map.max().item()  # actual max (corners ~45)
        self.r_map = r_map  # lazy .to(device)
        counts = torch.zeros(self.max_r + 1, dtype=torch.float32)
        counts.scatter_add_(0, r_map.flatten(), torch.ones(ps*ps, dtype=torch.float32))
        self.radial_counts = counts  # lazy .to(device)

        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    # ── Public interface ──

    def extract_single(self, pil_img):
        return self.extract_from_pil_list([pil_img])

    def extract_from_pil_list(self, pil_imgs, n_workers=4):
        rgb_t = torch.stack([self.img_transform(img.convert("RGB")) for img in pil_imgs]).to(self.device)
        gray_t = rgb_t.mean(dim=1, keepdim=True)

        ch0 = self._ch0_unet_batch(pil_imgs)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            ch1_list = list(pool.map(self._ch1_jpeg, pil_imgs))

        ch2 = self._ch2_phase_coherence_gpu(gray_t)
        ch3 = self._ch3_noise_model_gpu(gray_t)
        ch4 = self._ch4_srm_cross_channel_gpu(rgb_t)
        ch5 = self._ch5_spectral_deviation_gpu(gray_t)
        ch6 = self._ch6_vae_reconstruction_gpu(rgb_t)

        # Ch1 自适应抑制: 全图高压缩时降低 Ch1 信号 (避免全图假阳性)
        ch1_t = torch.from_numpy(np.stack(ch1_list)).float()
        ch1_mean = ch1_t.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        suppress = torch.where(ch1_mean > 0.15,
                               torch.clamp(0.15 / (ch1_mean + 1e-8), 0.2, 1.0),
                               torch.ones_like(ch1_mean))
        ch1_t = ch1_t * suppress

        return torch.stack([
            torch.from_numpy(np.stack(ch0)).float(),
            ch1_t,
            ch2.cpu().float(), ch3.cpu().float(),
            ch4.cpu().float(), ch5.cpu().float(),
            ch6.cpu().float(),
        ], dim=1)  # [B, 7, out, out]

    # ── Ch6: VAE Reconstruction Error (GPU batch, Flux VAE) ──

    @torch.no_grad()
    def _ch6_vae_reconstruction_gpu(self, rgb_t):
        """
        Ch6: VAE Reconstruction Error — DIRE 轻量化
        Flux VAE: encode(mode) → decode → |original - reconstructed| L1 均值

        与其他通道的互补性分析:
        - Ch0-Ch4 均为高频纹理/统计信号 → Ch6 用低频重建误差与之互补
        - 对 Doubao (0.62) > Flux (0.58) > SDXL (0.51): VAE 跨模型检测信号
        - 注意: 仅用自家 VAE 时对其他生成器检测能力有限 (DIRE 的已知局限)
        - 这里保持纯 L1 误差以维持与其他通道的正交互补性 (联合 AUC 最优)
        """
        B = rgb_t.shape[0]
        if self.vae is None:
            return torch.zeros(B, self.out_size, self.out_size, device=self.device)

        x = (rgb_t * 2.0 - 1.0).half()  # [-1, 1], fp16

        max_vae_batch = 8  # [V17 Perf] 20GB VRAM → 可放心提高
        error_maps = []
        for start in range(0, B, max_vae_batch):
            xb = x[start:start + max_vae_batch]
            latent_dist = self.vae.encode(xb).latent_dist
            latent = latent_dist.mode()  # 确定性重建，保持与其他通道的互补性
            recon = self.vae.decode(latent).sample.clamp(-1, 1)
            err = (xb.float() - recon.float()).abs().mean(dim=1, keepdim=True)  # [b, 1, 512, 512]
            error_maps.append(err)

        error = torch.cat(error_maps, dim=0)  # [B, 1, 512, 512]

        patches = F.unfold(error, self.patch_size, stride=self.stride)  # [B, ps*ps, N]
        scores = patches.mean(1) + patches.std(1)
        n = self._n_hw
        out = F.interpolate(scores.reshape(B, 1, n, n),
                            (self.out_size, self.out_size),
                            mode="bilinear", align_corners=False).squeeze(1)
        return out

    # ── Helpers ──

    def _unfold_patches(self, x):
        B = x.shape[0]
        p = F.unfold(x, self.patch_size, stride=self.stride)  # [B, ps*ps, N]
        N = p.shape[2]
        return p.permute(0, 2, 1).reshape(B * N, self.patch_size, self.patch_size), B, N

    def _to_out_map(self, scores, B, N):
        n = self._n_hw
        return F.interpolate(scores.reshape(B, 1, n, n).float(),
                             (self.out_size, self.out_size),
                             mode="bilinear", align_corners=False).squeeze(1)

    @staticmethod
    def _batch_pearson(a, b):
        # a, b: [B, ps*ps, N] → [B*N]
        a = a.permute(0, 2, 1); b = b.permute(0, 2, 1)  # [B, N, ps*ps]
        ac = a - a.mean(2, keepdim=True)
        bc = b - b.mean(2, keepdim=True)
        num = (ac * bc).sum(2)
        den = (ac.pow(2).sum(2) * bc.pow(2).sum(2)).sqrt() + 1e-8
        return (num / den).reshape(-1)  # [B*N]

    # ── Ch0: UNet 去噪残差 (GPU batch, 多σ探测 + 随机噪声) ──

    @torch.no_grad()
    def _ch0_unet_batch(self, pil_imgs):
        """
        RGB 多尺度去噪残差 v2:
        - UNet 输入 3ch RGB → 输出 3ch 噪声预测 → 跨通道残差融合
        - 3 个 σ 探测 (10, 25, 40) /255, 每张图使用图像相关的随机噪声
        - 跨通道一致性: R/G/B 残差的方差 → AI 生成图跨通道噪声更均匀
        - 自动兼容旧版 1ch UNet
        """
        if self.unet is None:
            return [np.zeros((self.out_size, self.out_size), np.float32) for _ in pil_imgs]

        in_ch = self.unet.in_ch

        if in_ch == 3:
            # RGB 模式
            imgs = np.stack([np.array(img.convert("RGB").resize((256, 256), Image.LANCZOS), np.float32) / 255.0
                             for img in pil_imgs])  # [B, 256, 256, 3]
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(self.device)  # [B, 3, 256, 256]
        else:
            # 旧版灰度兼容
            imgs = np.stack([np.array(img.convert("L").resize((256, 256), Image.LANCZOS), np.float32) / 255.0
                             for img in pil_imgs])
            x = torch.from_numpy(imgs).unsqueeze(1).to(self.device)  # [B, 1, 256, 256]

        B = x.shape[0]
        sigmas = [10.0 / 255.0, 25.0 / 255.0, 40.0 / 255.0]
        multi_residuals = []

        for sigma in sigmas:
            seeds = [int(x[i].mean().item() * 1e8) % (2**31) for i in range(B)]
            noise_list = []
            for i in range(B):
                g = torch.Generator(device=self.device).manual_seed(abs(seeds[i]))
                noise_list.append(torch.randn(1, in_ch, 256, 256, device=self.device, generator=g))
            noise = torch.cat(noise_list, dim=0) * sigma

            noisy = torch.clamp(x + noise, 0, 1)
            pred = self.unet(noisy)

            residual = torch.abs(pred - noise)  # [B, in_ch, 256, 256]
            multi_residuals.append(residual)

        # 融合: [B, 3*in_ch, 256, 256] → 跨σ + 跨通道
        stacked = torch.cat(multi_residuals, dim=1)  # [B, 3*in_ch, 256, 256]

        if in_ch == 3:
            # 跨σ均值+方差 (per RGB channel)
            s_reshaped = stacked.reshape(B, 3, in_ch, 256, 256)  # [B, n_sigma, 3ch, H, W]
            sigma_fused = s_reshaped.mean(dim=1) + s_reshaped.std(dim=1)  # [B, 3, 256, 256]
            # 跨通道融合: mean + 跨通道方差 (AI 图 RGB 噪声更均匀 → 方差小)
            fused = sigma_fused.mean(dim=1, keepdim=True) + sigma_fused.std(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            fused = stacked.mean(dim=1, keepdim=True) + stacked.std(dim=1, keepdim=True)

        res512 = F.interpolate(fused, (512, 512), mode="bilinear", align_corners=False)
        patches = F.unfold(res512, self.patch_size, stride=self.stride)
        scores = patches.mean(1) + patches.std(1)
        n = self._n_hw
        out = F.interpolate(scores.reshape(B, 1, n, n), (self.out_size, self.out_size),
                            mode="bilinear", align_corners=False).squeeze(1)
        return [out[i].cpu().numpy() for i in range(B)]

    # ── Ch1: JPEG (CPU, called via ThreadPool) ──

    def _ch1_jpeg(self, pil_img, quality_levels=(50, 75, 95)):
        img = pil_img.convert("RGB")
        arr = np.array(img, np.float32)
        residuals = []
        for q in quality_levels:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q); buf.seek(0)
            residuals.append(np.abs(arr.mean(2) - np.array(Image.open(buf).convert("RGB"), np.float32).mean(2)))
        residuals = np.stack(residuals)
        fused = residuals.mean(0) + 0.5 * np.abs(residuals[0] - residuals[-1])
        t = torch.from_numpy(fused).float().unsqueeze(0).unsqueeze(0)
        t512 = F.interpolate(t, (512, 512), mode="bilinear", align_corners=False)
        patches = F.unfold(t512, self.patch_size, stride=self.stride)
        scores = patches.mean(1) + patches.std(1)
        n = self._n_hw
        return F.interpolate(scores.reshape(1,1,n,n), (self.out_size, self.out_size),
                             mode="bilinear", align_corners=False).squeeze().numpy()

    # ── Ch2: 相位一致性 (GPU 全向量化) ──

    def _ch2_phase_coherence_gpu(self, gray_t):
        patches, B, N = self._unfold_patches(gray_t)
        phase = torch.angle(torch.fft.fft2(patches))
        dphi_y = phase[:, 1:, :] - phase[:, :-1, :]
        dphi_x = phase[:, :, 1:] - phase[:, :, :-1]
        vy = (dphi_y.sin().mean((1,2)).pow(2) + dphi_y.cos().mean((1,2)).pow(2)).sqrt()
        vx = (dphi_x.sin().mean((1,2)).pow(2) + dphi_x.cos().mean((1,2)).pow(2)).sqrt()
        return self._to_out_map(1.0 - (vy + vx) / 2.0, B, N)

    # ── Ch3: 噪声模型偏差 (GPU 全向量化) ──

    def _ch3_noise_model_gpu(self, gray_t):
        lp_k = torch.ones(1, 1, 3, 3, device=self.device) / 9.0
        smooth = F.conv2d(F.pad(gray_t, (1,1,1,1), mode="reflect"), lp_k)
        noise_sq = (gray_t - smooth).pow(2)
        pb = F.unfold(smooth,   self.patch_size, stride=self.stride)
        pn = F.unfold(noise_sq, self.patch_size, stride=self.stride)
        B, _, N = pb.shape
        return self._to_out_map(1.0 - self._batch_pearson(pb, pn), B, N)

    # ── Ch4: SRM 跨通道一致性 (GPU Conv2d + 向量化相关) ──

    def _ch4_srm_cross_channel_gpu(self, rgb_t):
        def apply_srm(ch):
            r = F.conv2d(F.pad(ch, (2,2,2,2), mode="reflect"), self.srm_kernel_t)
            return r.abs().mean(1, keepdim=True)
        sr = apply_srm(rgb_t[:, 0:1])
        sg = apply_srm(rgb_t[:, 1:2])
        sb = apply_srm(rgb_t[:, 2:3])
        pr = F.unfold(sr, self.patch_size, stride=self.stride)
        pg = F.unfold(sg, self.patch_size, stride=self.stride)
        pb = F.unfold(sb, self.patch_size, stride=self.stride)
        B, _, N = pr.shape
        avg_corr = (self._batch_pearson(pr, pg) + self._batch_pearson(pg, pb) +
                    self._batch_pearson(pr, pb)) / 3.0
        return self._to_out_map(1.0 - avg_corr, B, N)

    # ── Ch5: 频谱偏差 (GPU scatter_add 径向均值 + batched lstsq) ──

    def _ch5_spectral_deviation_gpu(self, gray_t):
        patches, B, N = self._unfold_patches(gray_t)
        patches = patches - patches.mean((1,2), keepdim=True)
        psd = torch.fft.fftshift(torch.fft.fft2(patches).abs().pow(2), dim=(-2,-1))

        r_map = self.r_map.to(self.device)
        r_flat = r_map.flatten()
        counts = self.radial_counts.to(self.device).clamp(min=1)

        psd_flat = psd.reshape(B * N, -1)
        radial_sum = torch.zeros(B * N, self.max_r + 1, device=self.device)
        radial_sum.scatter_add_(1, r_flat.unsqueeze(0).expand(B * N, -1), psd_flat)
        r_hi = self.inscribed_r - 1   # 31 — stay inside valid circle
        rp = (radial_sum / counts)[:, 2:r_hi]  # [B*N, n_r]

        valid = (rp > 0).all(1)
        log_s = rp.clamp(min=1e-10).log()

        freqs = torch.arange(2, r_hi, dtype=torch.float32, device=self.device)
        log_f = freqs.log()
        A = torch.stack([log_f, torch.ones_like(log_f)], dim=1)  # [n_r, 2]
        AtA_inv = torch.linalg.inv(A.T @ A)
        x = log_s @ A @ AtA_inv                                    # [B*N, 2]
        fitted = log_f * x[:, 0:1] + x[:, 1:2]
        residual = ((log_s - fitted).pow(2).mean(1)).sqrt()
        residual[~valid] = 0.0

        return self._to_out_map(residual, B, N)
