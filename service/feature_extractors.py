import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedConv(nn.Module):
    """Fixed (non-trainable) convolution layer for handcrafted filters."""
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use explicit padding for broader PyTorch compatibility.
        kh, kw = int(self.weight.shape[-2]), int(self.weight.shape[-1])
        pad_h = kh // 2
        pad_w = kw // 2
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        return F.conv2d(x, self.weight, padding=0)


class MultiFeatureExtractor(nn.Module):
    """
    Lightweight handcrafted feature extractor for frequency, noise residual, and edge cues.
    Returns a small tensor aligned to the CLIP block3 resolution (default 28x28).
    """
    def __init__(self, target_size: int = 28):
        super().__init__()
        self.target_size = target_size

        # High-pass kernel for noise residual (Laplacian)
        hp = torch.tensor([
            [0.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Sobel kernels for edge gradients
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.noise_conv = FixedConv(hp)
        self.sobel_x = FixedConv(sobel_x)
        self.sobel_y = FixedConv(sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: normalized image tensor [B, 3, H, W] (CLIP-normalized input is fine).
        Returns:
            features: [B, 3, target_size, target_size] (freq, noise, edge magnitude).
        """
        # Force FP32 for numerical stability / torch.fft compatibility under AMP.
        x = x.float()
        b, _, h, w = x.shape

        # Convert to grayscale for handcrafted cues
        gray = x.mean(dim=1, keepdim=True)

        # Frequency magnitude via FFT (log scaled)
        freq = torch.fft.fft2(gray)
        freq = torch.abs(freq)
        freq = torch.log1p(freq)
        freq = F.adaptive_avg_pool2d(freq, self.target_size)
        freq = self._safe_normalize(freq)

        # Noise residual (high-pass)
        noise = torch.abs(self.noise_conv(gray))
        noise = F.adaptive_avg_pool2d(noise, self.target_size)
        noise = self._safe_normalize(noise)

        # Edge magnitude from Sobel gradients
        grad_x = self.sobel_x(gray)
        grad_y = self.sobel_y(gray)
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edges = F.adaptive_avg_pool2d(edges, self.target_size)
        edges = self._safe_normalize(edges)

        feats = torch.cat([freq, noise, edges], dim=1)
        return feats

    def _safe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        std = torch.clamp(std, min=1e-6)
        return (x - mean) / std
