"""
models/compartment.py
=====================
E4 — Compartment Branches.

Three crops (global knee, medial, lateral) share a single ConvNeXt-tiny
backbone — medial and lateral branches additionally share weights with
each other.  An Edge-Gated Residual Block (EGRB) preserves fine joint
margin activations before pooling.  A gated CompartmentFusion produces
one 768-d fused descriptor while also exposing the per-branch pooled
vectors for downstream RTC (E7).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from models.dual_intensity import DualIntensityStem


class EdgeGatedResidualBlock(nn.Module):
    """
    Edge-Gated Residual Block (EGRB).

    A directional Sobel gate (4-direction, grayscale-averaged) is computed
    from the input feature map and applied multiplicatively to the main
    depthwise convolution branch, preserving fine structural activations
    at joint margins.

    ``output = main_branch(x) ⊗ sobel_gate(x) + x``

    Parameters
    ----------
    channels:
        Number of input (= output) channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw_conv   = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.pw_conv   = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn        = nn.BatchNorm2d(channels)
        self.act       = nn.GELU()
        self.edge_proj = nn.Sequential(
            nn.Conv2d(4, channels, 1, bias=False), nn.Sigmoid()
        )
        _sobel = {
            "kH":  [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "kV":  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "kD1": [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            "kD2": [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        }
        for name, k in _sobel.items():
            self.register_buffer(
                name, torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)
            )

        # Identity-preserving init (mirrors KneeLocalizer's explicit
        # identity init in models/localization.py, which this block lacked).
        # Zero-initing pw_conv makes main=0 at step 0, so
        # forward() = main*gate + x == x regardless of the gate's value —
        # EGRB starts as a true identity function instead of injecting
        # untrained conv noise into medial/lateral features from epoch 1.
        nn.init.zeros_(self.pw_conv.weight)

    def _sobel_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Mean Sobel response across channels → (B, 4, H, W)."""
        g = x.mean(dim=1, keepdim=True)
        return torch.cat([
            F.conv2d(g, self.kH,  padding=1),
            F.conv2d(g, self.kV,  padding=1),
            F.conv2d(g, self.kD1, padding=1),
            F.conv2d(g, self.kD2, padding=1),
        ], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C, H, W)."""
        main = self.act(self.bn(self.pw_conv(self.dw_conv(x))))
        gate = self.edge_proj(self._sobel_gate(x))
        return main * gate + x


class _EncoderBranch(nn.Module):
    """
    Single-crop encoder: stem → shared backbone features → EGRB → pool.

    Both the stem and backbone_features references are injected, not
    owned, so the caller controls weight sharing between branches.

    Parameters
    ----------
    stem:
        DualIntensityStem (or nn.Identity if E3 is disabled).
    backbone_features:
        Shared ConvNeXt-tiny feature extractor (nn.Sequential).
    feature_dim:
        Backbone output channels (768 for convnext_tiny).
    """

    def __init__(
        self,
        stem: nn.Module,
        backbone_features: nn.Module,
        feature_dim: int = 768,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.stem            = stem
        self.features        = backbone_features
        self.grad_checkpoint = grad_checkpoint
        self.egrb            = EdgeGatedResidualBlock(feature_dim)
        self.pool            = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, feature_dim)."""
        # Gradient checkpointing: active only during training when explicitly
        # enabled — eval() runs are byte-identical to the no-flag default.
        if self.training and self.grad_checkpoint:
            spatial = torch.utils.checkpoint.checkpoint(
                self.features, self.stem(x), use_reentrant=False
            )
        else:
            spatial = self.features(self.stem(x))
        return self.pool(self.egrb(spatial)).flatten(1)


class CompartmentFusion(nn.Module):
    """
    Soft-gated fusion of global, medial, and lateral branch features.

    A 3-way softmax gate learns how much each branch contributes to the
    fused representation, preventing any single branch from dominating.

    ``fused = w_g · global + w_m · medial + w_l · lateral``

    Parameters
    ----------
    feat_dim:
        Per-branch backbone feature dimension.
    """

    def __init__(self, feat_dim: int = 768) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 3, 3), nn.Softmax(dim=-1)
        )
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.GELU()
        )

        # Bias the gate toward "mostly global" at init instead of the
        # ~uniform (1/3, 1/3, 1/3) softmax a zero-init linear produces.
        # Medial/lateral start as EGRB-identity passthroughs of raw backbone
        # features (see EdgeGatedResidualBlock's init above), but the fused
        # *combination* was still ~2/3 medial+lateral vs 1/3 global at step
        # 0 — a large, sudden departure from E1's known-good global-only
        # representation for the classifier to absorb before any of the new
        # branches have learned anything useful. Softmax([3,0,0]) ≈
        # (0.85, 0.07, 0.07): close to E1's behavior initially, with the
        # branches' influence growing only as the gate is trained.
        nn.init.zeros_(self.gate[0].weight)
        with torch.no_grad():
            self.gate[0].bias.copy_(torch.tensor([3.0, 0.0, 0.0]))

    def forward(
        self,
        global_feat: torch.Tensor,
        medial_feat: torch.Tensor,
        lateral_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Three (B, feat_dim) tensors → one (B, feat_dim) tensor."""
        concat = torch.cat([global_feat, medial_feat, lateral_feat], dim=1)
        w = self.gate(concat)
        if self.training:
            self.debug_stats = {
                "gate_global": w[:, 0].mean().item(),
                "gate_medial": w[:, 1].mean().item(),
                "gate_lateral": w[:, 2].mean().item(),
            }
        norm_shape = (self.feat_dim,)
        global_feat = F.layer_norm(global_feat, norm_shape)
        medial_feat = F.layer_norm(medial_feat, norm_shape)
        lateral_feat = F.layer_norm(lateral_feat, norm_shape)
        fused = w[:, 0:1] * global_feat + w[:, 1:2] * medial_feat + w[:, 2:3] * lateral_feat
        return self.proj(fused)


class CompartmentBranchModule(nn.Module):
    """
    E4 top-level module — shared-weight three-crop encoder.

    The global, medial, and lateral crops all pass through the **same**
    DualIntensityStem and ConvNeXt-tiny backbone_features weights.
    Medial and lateral branches additionally share their EGRB weights.

    Parameters
    ----------
    stem:
        Injected DualIntensityStem (or nn.Identity).
    backbone_features:
        Injected shared ConvNeXt-tiny feature extractor.
    feature_dim:
        Backbone output channel count.

    Returns (from forward)
    ----------------------
    fused_feat:  (B, feature_dim) — gated fusion of all three branches
    global_feat: (B, feature_dim) — global branch (→ E7 RTC, E5 DRP)
    medial_feat: (B, feature_dim) — medial branch (→ E7 RTC)
    lateral_feat:(B, feature_dim) — lateral branch (→ E7 RTC)
    """

    def __init__(
        self,
        stem: nn.Module,
        backbone_features: nn.Module,
        feature_dim: int = 768,
        overlap: float = 0.10,
        debug_visualization: bool = False,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.overlap = overlap
        self.debug_visualization = debug_visualization
        self.compartment_branch = _EncoderBranch(
            stem,
            backbone_features,
            feature_dim,
            grad_checkpoint=grad_checkpoint,
        )
        self.fusion = CompartmentFusion(feature_dim)

    def _split_compartments(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split a full knee image into overlapping medial and lateral crops.
        """

        _, _, h, w = x.shape

        overlap = int(self.overlap * w)
        mid = w // 2

        medial = x[:, :, :, :mid + overlap]
        lateral = x[:, :, :, mid - overlap:]

        medial = F.interpolate(
            medial,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        lateral = F.interpolate(
            lateral,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        return medial, lateral

    def forward(
        self,
        global_feat,
        global_crop,
    ):
        """
        Fuse the global feature with medial and lateral compartment features.
        Returns:
            fused_feat, medial_feat, lateral_feat
        """
        medial_crop, lateral_crop = self._split_compartments(global_crop)
        if (
            self.training
            and self.debug_visualization
            and not hasattr(self, "_crop_debug")
        ):
            self._crop_debug = True

            from pathlib import Path
            import matplotlib.pyplot as plt

            Path("results/debug").mkdir(parents=True, exist_ok=True)

            images = {
                "global": global_crop[0],
                "medial": medial_crop[0],
                "lateral": lateral_crop[0],
            }

            # Undo ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            for name, img in images.items():
                img = img.detach().cpu()

                mean_cpu = mean.cpu()
                std_cpu = std.cpu()

                img = img * std_cpu + mean_cpu
                img = img.clamp(0, 1)

                plt.figure(figsize=(5, 5))
                plt.imshow(img.permute(1, 2, 0))
                plt.axis("off")
                plt.title(name.capitalize())
                plt.savefig(f"results/debug/{name}.png", dpi=200, bbox_inches="tight")
                plt.close()
        m = self.compartment_branch(medial_crop)
        l = self.compartment_branch(lateral_crop)

        fused = self.fusion(
            global_feat,
            m,
            l,
        )
        if self.training:
            self.debug_stats = {
                "global_norm": global_feat.norm(dim=1).mean().item(),
                "medial_norm": m.norm(dim=1).mean().item(),
                "lateral_norm": l.norm(dim=1).mean().item(),

                "gm_diff": (global_feat - m).abs().mean().item(),
                "gl_diff": (global_feat - l).abs().mean().item(),
                "ml_diff": (m - l).abs().mean().item(),
            }
        if self.training:
            self.debug_stats.update(self.fusion.debug_stats)
        return fused, m, l
