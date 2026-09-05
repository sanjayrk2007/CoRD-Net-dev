"""
config.py
=========
Typed configuration dataclasses for CoRD-Net.

All hyperparameters live here.  No magic numbers appear elsewhere in the
codebase — every module receives a config object via dependency injection.

Usage
-----
    from config import ModelConfig, TrainingConfig, get_config

    cfg = get_config("e8")
    model = DRPNet(cfg.model)
    trainer = Trainer(model, cfg.training)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Complete specification of the DRPNet architecture."""

    # Backbone
    backbone: str = "convnext_tiny"
    pretrained: bool = False
    backbone_feature_dim: int = 768   # ConvNeXt-tiny pooled output
    spatial_feature_dim: int = 768    # ConvNeXt-tiny spatial (before pool)

    # Embedding dimensions
    embedding_dim: int = 256          # DRP / PGR / RTC shared dim
    fused_dim: int = 512              # after projecting concatenated feats
    metric_embed_dim: int = 128       # MetricEmbeddingHead (SupCon)

    # Dataset
    num_classes: int = 5
    image_size: int = 224             # backbone canonical size
    in_channels: int = 3

    # STN (E2)
    stn_img_size: int = 299

    # Compartments (E4)
    compartment_overlap: float = 0.10

    # PGR (E6)
    prototype_temperature: float = 0.07
    prototype_ema_momentum: float = 0.99
    pgr_num_heads: int = 4
    pgr_dropout: float = 0.1

    # RTC (E7)
    rtc_num_heads: int = 4
    rtc_dropout: float = 0.1
    rtc_use_global_context: bool = True

    # FGBF Module Parameters
    fgbf_feature_dim: int = 256
    fgbf_hidden_dim: int = 128
    fgbf_dropout: float = 0.1
    fgbf_loss_weight: float = 0.15
    fgbf_fuse_main: bool = False
    fgbf_block: str = "baseline"

    skip_fixed_clahe_if_dual_intensity: bool = True
    kaggle_priority: Optional[int] = None

    # Ablation flags — set by get_config(experiment)
    use_stn: bool = False
    use_dual_intensity: bool = False
    use_fgbf: bool = False
    use_compartment: bool = False
    use_drp: bool = False
    use_pgr: bool = False
    use_rtc: bool = False
    use_aux_heads: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training loop, optimiser, and scheduler settings."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"         # 'cosine' | 'step' | 'none'
    warmup_epochs: int = 5

    batch_size: int = 16
    epochs: int = 100

    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "h1": 1.0, "h2": 0.5, "h3": 0.3,
        "h4": 0.4, "h5": 0.4, "h6": 0.3,
        "h7": 0.2, "proto": 0.3,
    })

    active_heads: List[str] = field(
        default_factory=lambda: ["h1", "h2", "h3", "h4", "h5", "h6", "h7"]
    )

    device: Optional[str] = None      # None = auto-detect
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    gradient_clip: float = 1.0
    amp: bool = False

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    save_every: int = 10
    patience: Optional[int] = None    # None = disable early stopping

    sampler: str = "none"            # 'none' | 'weighted'
    sampler_power: float = 1.0       # 1.0=full inverse-freq, 0.5=sqrt-softened, 0.0=uniform.
                                      # Use < 1.0 when loss_type is already 'weighted_ce' to
                                      # avoid double-correcting the same imbalance.
    augmentation: str = "mild"       # 'standard' | 'mild' | 'none'
    loss_type: str = "ce"            # 'ce' | 'weighted_ce' | 'focal' | 'soft_qwk' | 'ce_qwk' | 'boundary_aware'
    # 'qwk' | 'kl1_only' | 'score'. Historical note: this field was declared
    # but never read by trainer.py before this patch, so every experiment run
    # so far (including e2_fgbf_pim_v2) actually used the 'kl1_only' formula
    # regardless of this value. Default corrected to match that reality.
    checkpoint_monitor: str = "kl1_only"
    min_epochs_before_early_stop: int = 15
    low_grade_recall_floor: float = 0.30  # checkpoint guard (monitor="score" only):
                                           # don't save "best" if the weakest of
                                           # KL0/KL1/KL2 recall drops below this
    stn_identity_reg_weight: float = 0.0   # STN affine matrix identity regularization weight (0.0 = disabled)

    # ── Dataset paths (set via CLI; no hardcoded paths) ───────────────────
    data_root: Optional[str] = None
    """Root directory of the OAI dataset (required for real training)."""

    metadata_csv: Optional[str] = None
    """Path to OAI metadata CSV with KL/JSN/osteophyte labels.
    If None, KL grade is inferred from the subdirectory name and
    auxiliary labels default to -1 (ignored by loss)."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    medial_suffix: str = "_MED"
    lateral_suffix: str = "_LAT"
    """Filename suffixes used to derive compartment crop paths.
    E.g. image "001.png" → medial "001_MED.png", lateral "001_LAT.png".
    Override if your OAI layout uses different conventions."""

    # ── Split Mode 3: separate CSV per split ──────────────────────────────
    train_csv: Optional[str] = None
    """Path to a CSV containing only training samples (Mode 3)."""

    val_csv: Optional[str] = None
    """Path to a CSV containing only validation samples (Mode 3)."""

    test_csv: Optional[str] = None
    """Path to a CSV containing only test samples (Mode 3).
    Optional — if omitted, the test DataLoader will be empty."""


# ──────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ──────────────────────────────────────────────────────────────────────────────

# Reusable flags fragment for validated FGBF module
FGBF_FLAGS: Dict[str, any] = {
    "use_fgbf": True,
    "fgbf_block": "pim",
    "fgbf_fuse_main": True,
}

# Note: Intermediate ablation/debugging keys (e2m, e1_fgbf, e2_fgbf, e2_fgbf_ms,
# e2_fgbf_sk, e2_fgbf_pim_v2-v5, e6, e7, e8) have been trimmed from the active
# registry and are recoverable from git log history for config.py if needed.
_EXPERIMENT_FLAGS: Dict[str, Tuple[str, Dict[str, any]]] = {
    "e1": (
        "Baseline ConvNeXt",
        {
            "kaggle_priority": 1,
        }
    ),

    "e2": (
        "ConvNeXt + Auto-Localization (STN)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "kaggle_priority": None,
        }
    ),

    "e3": (
        "E2 + Dual-Intensity Stem",
        {
            "use_stn": True,
            "use_dual_intensity": True,
            "kaggle_priority": None,
        }
    ),

    "e3_fgbf": (
        "E3 + Fine-Grained Boundary Feature Module",
        {
            "use_stn": True,
            "use_dual_intensity": True,
            **FGBF_FLAGS,
            "kaggle_priority": 2,
        }
    ),

    "e2_fgbf_pim": (
        "E2 + FGBF + PIM-Lite Feature Block",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "pim",
            "fgbf_fuse_main": False,
            "kaggle_priority": None,
        }
    ),

    "e2_fgbf_pim_v6": (
        "E2 + FGBF + PIM-Lite Feature Block (Boundary-Aware Loss + STN Reg)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
            "kaggle_priority": 3,
        }
    ),

    "e2_fgbf_cbam": (
        "E2 + FGBF + CBAM-Lite Control Block",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "cbam",
            "kaggle_priority": None,
        }
    ),

    "e4": (
        "E2 + Compartment Branches",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_compartment": True,
            **FGBF_FLAGS,
            "kaggle_priority": None,
        }
    ),

    "e5": (
        "E4 + DRP Block",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_compartment": True,
            "use_drp": True,
            **FGBF_FLAGS,
            "kaggle_priority": None,
        }
    ),
}

EXPERIMENT_NAMES: Dict[str, str] = {k: v[0] for k, v in _EXPERIMENT_FLAGS.items()}


@dataclass
class Config:
    """Top-level config bundling model + training settings."""
    experiment: str
    model: ModelConfig
    training: TrainingConfig

    @property
    def description(self) -> str:
        return EXPERIMENT_NAMES.get(self.experiment, self.experiment)


def get_config(
    experiment: str,
    *,
    pretrained: bool = False,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    data_root: Optional[str] = None,
    metadata_csv: Optional[str] = None,
) -> Config:
    """Return a fully-merged Config for *experiment*."""
    if experiment not in _EXPERIMENT_FLAGS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Valid choices: {list(_EXPERIMENT_FLAGS.keys())}"
        )
    _, flags = _EXPERIMENT_FLAGS[experiment]
    model_cfg = ModelConfig(pretrained=pretrained, **flags)
    train_cfg = TrainingConfig()

    # From e3 onward (except untuned ablations), default to class-balanced loss and sampler
    if experiment in ("e3", "e3_fgbf", "e4", "e5"):
        train_cfg.loss_type = "weighted_ce"
        train_cfg.sampler = "weighted"
        train_cfg.sampler_power = 0.5
        train_cfg.checkpoint_monitor = "score"

    # e2_fgbf_pim_v6: tuned recipe (BoundaryAwareLoss, STN identity reg, stabilized LR and warmup)
    if experiment == "e2_fgbf_pim_v6":
        train_cfg.loss_type = "boundary_aware"
        train_cfg.sampler = "weighted"
        train_cfg.sampler_power = 0.5
        train_cfg.checkpoint_monitor = "score"
        train_cfg.stn_identity_reg_weight = 0.015
        train_cfg.learning_rate = 5e-5
        train_cfg.warmup_epochs = 8

    if device is not None:
        train_cfg.device = device
    if batch_size is not None:
        train_cfg.batch_size = batch_size
    if epochs is not None:
        train_cfg.epochs = epochs
    if learning_rate is not None:
        train_cfg.learning_rate = learning_rate
    if data_root is not None:
        train_cfg.data_root = data_root
    if metadata_csv is not None:
        train_cfg.metadata_csv = metadata_csv
    return Config(experiment=experiment, model=model_cfg, training=train_cfg)
