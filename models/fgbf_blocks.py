"""
models/fgbf_blocks.py
=====================
Lightweight Fine-Grained Feature Refinement Blocks for FGBF.

Designed to enhance feature discrimination between low-grade osteoarthritis
classes (KL0, KL1, KL2) inside the FGBF module.

All blocks:
- Accept input tensors of shape (B, 256, H, W).
- Output tensors of shape (B, 256, H, W).
- Preserve spatial height (H) and width (W).
- Employ residual refinement (output = input + refined_feature).
- Are lightweight and compatible with 8 GB VRAM (e.g. RTX 5050).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFineGrainedBlock(nn.Module):
    """
    Lightweight Multi-Scale Feature Refinement Block.

    Uses multiple receptive fields via dilated 3x3 convolutions and a context branch,
    concatenates branch outputs, and fuses them back to 256 channels via 1x1 convolution
    with residual refinement.

    Branches (each width = 64):
    - Branch 1: 3x3 Conv, dilation=1, padding=1
    - Branch 2: 3x3 Conv, dilation=2, padding=2
    - Branch 3: 3x3 Conv, dilation=3, padding=3
    - Branch 4: 5x5 Conv, padding=2 (lightweight context branch)
    """

    def __init__(self, channels: int = 256, branch_width: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.branch_width = branch_width

        # Local branch (3x3, dilation=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, branch_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_width),
            nn.GELU(),
        )

        # Medium-context branch (3x3, dilation=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, branch_width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_width),
            nn.GELU(),
        )

        # Larger-context branch (3x3, dilation=3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, branch_width, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(branch_width),
            nn.GELU(),
        )

        # Lightweight 5x5 context branch
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, branch_width, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(branch_width),
            nn.GELU(),
        )

        # Fusion projection: (64 * 4) = 256 -> 256
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_width * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        fused = self.fusion(concat)
        return self.act(x + fused)


class SelectiveKernelFineGrainedBlock(nn.Module):
    """
    Lightweight Selective Kernel-Style Feature Refinement Block.

    Uses two receptive-field branches (local 3x3 conv vs dilated 3x3 conv),
    extracts a global descriptor via adaptive pooling, predicts dynamic scale-selection
    weights for each branch, adaptively fuses the branches, and applies residual refinement.
    """

    def __init__(self, channels: int = 256, reduction: int = 8) -> None:
        super().__init__()
        self.channels = channels
        mid_channels = max(16, channels // reduction)

        # Branch A: Local 3x3 conv
        self.branch_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Branch B: Broader-context 3x3 conv (dilation=2)
        self.branch_b = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=32, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Global descriptor FC reduction
        self.fc_reduce = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
        )

        # Branch weight heads
        self.head_a = nn.Linear(mid_channels, channels)
        self.head_b = nn.Linear(mid_channels, channels)

        self.out_bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_a = self.branch_a(x)
        feat_b = self.branch_b(x)

        # Sum of branches for descriptor extraction
        feat_sum = feat_a + feat_b  # (B, C, H, W)
        s = feat_sum.mean(dim=(-2, -1))  # (B, C)

        z = self.fc_reduce(s)  # (B, mid_channels)

        w_a = self.head_a(z)  # (B, C)
        w_b = self.head_b(z)  # (B, C)

        # Softmax selection per channel between the two branches
        weights = torch.softmax(torch.stack([w_a, w_b], dim=0), dim=0)  # (2, B, C)
        w_a = weights[0].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        w_b = weights[1].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        fused = w_a * feat_a + w_b * feat_b
        fused = self.out_bn(fused)
        return self.act(x + fused)


class PIMLiteFineGrainedBlock(nn.Module):
    """
    Lightweight Position/Pixel Importance Module (PIM-Lite).

    Emphasizes informative local spatial regions using depthwise-pointwise feature
    extraction, lightweight spatial importance estimation, and residual refinement.
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.channels = channels

        # Local depthwise-pointwise feature extractor
        self.local_extract = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Spatial/pixel importance estimator
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Final pointwise projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.local_extract(x)  # (B, C, H, W)
        attn = self.spatial_attn(feat)  # (B, 1, H, W)
        refined = feat * attn  # (B, C, H, W)
        out = self.proj(refined)  # (B, C, H, W)
        return self.act(x + out)


class CBAMFineGrainedBlock(nn.Module):
    """
    Lightweight Convolutional Block Attention Module (CBAM-Lite Control).

    Applies sequential Channel Attention and Spatial Attention with residual refinement.
    """

    def __init__(self, channels: int = 256, reduction: int = 16) -> None:
        super().__init__()
        self.channels = channels
        reduced_dim = max(16, channels // reduction)

        # Channel Attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.GELU(),
            nn.Linear(reduced_dim, channels, bias=False),
        )

        # Spatial Attention Conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        avg_pool = x.mean(dim=(-2, -1))  # (B, C)
        max_pool = x.amax(dim=(-2, -1))  # (B, C)
        channel_attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).unsqueeze(-1).unsqueeze(-1)
        x_ca = x * channel_attn

        # Spatial Attention
        spatial_avg = x_ca.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_max = x_ca.amax(dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_input = torch.cat([spatial_avg, spatial_max], dim=1)  # (B, 2, H, W)
        spatial_attn = self.spatial_conv(spatial_input)  # (B, 1, H, W)
        x_sa = x_ca * spatial_attn

        out = self.bn(x_sa)
        return self.act(x + out)


def build_fgbf_block(block_type: str, channels: int = 256) -> nn.Module:
    """
    Factory function for instantiating fine-grained FGBF feature refinement blocks.

    Parameters
    ----------
    block_type:
        One of 'baseline', 'identity', 'none', 'multiscale', 'sk', 'pim', 'cbam'.
    channels:
        Number of feature channels (default: 256).

    Returns
    -------
    nn.Module:
        Instantiated block operating on (B, channels, H, W).
    """
    block_type_clean = block_type.strip().lower()
    if block_type_clean in ("baseline", "identity", "none"):
        return nn.Identity()
    elif block_type_clean in ("multiscale", "ms"):
        return MultiScaleFineGrainedBlock(channels=channels)
    elif block_type_clean in ("sk", "selective_kernel"):
        return SelectiveKernelFineGrainedBlock(channels=channels)
    elif block_type_clean in ("pim", "pim_lite"):
        return PIMLiteFineGrainedBlock(channels=channels)
    elif block_type_clean in ("cbam", "cbam_lite"):
        return CBAMFineGrainedBlock(channels=channels)
    else:
        raise ValueError(
            f"Unknown FGBF block type '{block_type}'. "
            f"Valid options: ['baseline', 'multiscale', 'sk', 'pim', 'cbam']"
        )
