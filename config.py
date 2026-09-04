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
    stn_img_size: int = 512

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

_EXPERIMENT_FLAGS: Dict[str, Tuple[str, Dict[str, any]]] = {
    "e1": (
        "Baseline ConvNeXt",
        {}
    ),

    "e2": (
        "ConvNeXt + Auto-Localization (STN)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
        }
    ),

    "e2m": (
        "E2 + Mild Augmentation",
        {
            "use_stn": True,
            "use_dual_intensity": False,
        }
    ),
    "e1_fgbf": (
        "E1 + Fine-Grained Boundary Feature Module",
        {
            "use_stn": False,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "baseline",
        }
    ),

    "e2_fgbf": (
        "E2 + Fine-Grained Boundary Feature Module",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "baseline",
        }
    ),

    "e2_fgbf_ms": (
        "E2 + FGBF + Multi-Scale Feature Block",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "multiscale",
        }
    ),

    "e2_fgbf_sk": (
        "E2 + FGBF + Selective Kernel Feature Block",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "sk",
        }
    ),

    "e2_fgbf_pim": (
        "E2 + FGBF + PIM-Lite Feature Block",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "pim",
        }
    ),

    "e2_fgbf_pim_v2": (
        "E2 + FGBF + PIM-Lite Feature Block (Fused)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
        }
    ),

    # v2 result: KL1 recall 0.44/0.42 (val/test) but KL2 recall regressed to
    # 0.43/0.39 (was ~0.60 unfused) — the sampler (raw inverse-freq) and
    # weighted_ce loss were both fully correcting the same imbalance at once,
    # over-boosting KL1 at its neighbors' expense. v3 isolates exactly two
    # changes vs v2: soften the sampler (sampler_power) and switch to the
    # guarded composite checkpoint monitor. fgbf_loss_weight and weight_decay
    # are deliberately left at v2's values so any change in outcome can be
    # attributed to these two fixes alone, not conflated with other knobs.
    "e2_fgbf_pim_v3": (
        "E2 + FGBF + PIM-Lite Feature Block (Fused, softened rebalance)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
        }
    ),

    # Follow-up only — run after v3, and only if v3 alone doesn't fully
    # resolve the KL1/KL2 trade-off. Adds weight_decay on top of v3 in
    # isolation so its effect isn't conflated with the sampler/monitor fix.
    "e2_fgbf_pim_v3b": (
        "E2 + FGBF + PIM-Lite Feature Block (v3 + higher weight decay)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
        }
    ),

    # v3 result: best_score peaked at epoch 4 (lr=6.4e-5, still ramping
    # through warmup) and was never beaten again in 20 further epochs.
    # Per-epoch log shows why: LR hits its cosine peak (1e-4) at epoch 6
    # and stays within ~10% of peak for the rest of the run (T_max=55 vs.
    # a patience-limited real run length of ~24 epochs — the schedule
    # never gets far enough into its decay to matter). Epoch 10 shows an
    # outright collapse (score 0.19, kappa 0.36) at near-peak LR, and
    # KL1 F1 *does* clear epoch 4's value more than once later on
    # (epochs 13/17/19), but always at KL0's or KL2's expense — the
    # model keeps sliding between class-biased optima instead of holding
    # a joint balance, consistent with LR staying too high for too long
    # rather than a validation-noise artifact. v3c isolates exactly two
    # changes vs v3: halve the peak LR and extend warmup so more of the
    # run happens in the gentler, epoch-4-like regime. sampler_power and
    # checkpoint_monitor are kept at v3's values so any change in outcome
    # is attributable to the LR schedule alone.
    "e2_fgbf_pim_v3c": (
        "E2 + FGBF + PIM-Lite Feature Block (v3 + stabilized LR schedule)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
        }
    ),

    # v4 isolates the combined class-weighted + soft-QWK loss (loss_type="ce_qwk")
    # on top of v3. All other settings (sampler_power=0.5, checkpoint_monitor="score",
    # LR=1e-4, warmup=5) are kept identical to v3 so the effect of the loss function
    # alone can be attributed without conflation.
    "e2_fgbf_pim_v4": (
        "E2 + FGBF + PIM-Lite Feature Block (v3 + combined CE/soft-QWK loss)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
        }
    ),

    # v5 isolates STN identity matrix regularization (stn_identity_reg_weight=0.01)
    # on top of v4. Penalizes deviation of the affine transform matrix theta from
    # identity [[1,0,0],[0,1,0]] to prevent aggressive or unstable spatial warping.
    "e2_fgbf_pim_v5": (
        "E2 + FGBF + PIM-Lite Feature Block (v4 + STN identity regularization)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
        }
    ),

    # v6 combines STN identity regularization with BoundaryAwareLoss (CE + SoftQWK + Grade Distance)
    # and stabilized learning rate (5e-5) with 8-epoch warmup.
    "e2_fgbf_pim_v6": (
        "E2 + FGBF + PIM-Lite Feature Block (Boundary-Aware Loss + STN Reg)",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            **FGBF_FLAGS,
        }
    ),

    "e2_fgbf_cbam": (
        "E2 + FGBF + CBAM-Lite Control Block",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_fgbf": True,
            "fgbf_block": "cbam",
        }
    ),

    # Separate ablation only
    "e3": (
        "E2 + Dual-Intensity Stem",
        {
            "use_stn": True,
            "use_dual_intensity": True,
            **FGBF_FLAGS,
        }
    ),

    "e3_fgbf": (
        "E3 + Fine-Grained Boundary Feature Module",
        {
            "use_stn": True,
            "use_dual_intensity": True,
            **FGBF_FLAGS,
        }
    ),

    # Main progression starts from E2, NOT E3
    "e4": (
        "E2 + Compartment Branches",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_compartment": True,
            **FGBF_FLAGS,
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
        }
    ),

    "e6": (
        "E5 + Prototype-Guided Refinement",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_compartment": True,
            "use_drp": True,
            "use_pgr": True,
            **FGBF_FLAGS,
        }
    ),

    "e7": (
        "E6 + Relational Token Coupling",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_compartment": True,
            "use_drp": True,
            "use_pgr": True,
            "use_rtc": True,
            **FGBF_FLAGS,
        }
    ),

    "e8": (
        "E7 + Auxiliary Heads",
        {
            "use_stn": True,
            "use_dual_intensity": False,
            "use_compartment": True,
            "use_drp": True,
            "use_pgr": True,
            "use_rtc": True,
            "use_aux_heads": True,
            **FGBF_FLAGS,
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
    """Return a fully-merged Config for *experiment* (e1 … e8, e2_fgbf, e3_fgbf)."""
    if experiment not in _EXPERIMENT_FLAGS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Valid choices: {list(_EXPERIMENT_FLAGS.keys())}"
        )
    _, flags = _EXPERIMENT_FLAGS[experiment]
    model_cfg = ModelConfig(pretrained=pretrained, **flags)
    train_cfg = TrainingConfig()

    # From e2_fgbf_pim_v2 onward, default to class-balanced loss and sampler
    if experiment in ("e2_fgbf_pim_v2", "e2_fgbf_pim_v3", "e2_fgbf_pim_v3b",
                       "e2_fgbf_pim_v3c", "e2_fgbf_pim_v4", "e2_fgbf_pim_v5",
                       "e2_fgbf_pim_v6", "e3", "e3_fgbf", "e4", "e5", "e6", "e7", "e8"):
        train_cfg.loss_type = "weighted_ce"
        train_cfg.sampler = "weighted"

    # v3: soften the sampler (avoid stacking two full corrections on the same
    # imbalance) and switch to the guarded composite monitor. Nothing else
    # changes vs v2 — see the registry comment above for why.
    if experiment in ("e2_fgbf_pim_v3", "e2_fgbf_pim_v3b", "e2_fgbf_pim_v3c",
                       "e2_fgbf_pim_v4", "e2_fgbf_pim_v5", "e2_fgbf_pim_v6"):
        train_cfg.sampler_power = 0.5
        train_cfg.checkpoint_monitor = "score"

    # v3b: isolated follow-up, only run if v3 needs more help.
    if experiment == "e2_fgbf_pim_v3b":
        train_cfg.weight_decay = 2e-4

    # v3c: isolated LR-stability follow-up — see registry comment above.
    # Halve peak LR and extend warmup from 5 -> 8 epochs so the model
    # spends more of its (patience-limited) real training window in the
    # gentler regime that produced v3's epoch-4 peak, instead of jumping
    # to and lingering at a peak LR the fused model can't hold a joint
    # class balance at.
    if experiment == "e2_fgbf_pim_v3c":
        train_cfg.learning_rate = 5e-5
        train_cfg.warmup_epochs = 8

    # v4: isolated combined CE + soft-QWK loss (loss_type="ce_qwk") vs v3.
    if experiment in ("e2_fgbf_pim_v4", "e2_fgbf_pim_v5"):
        train_cfg.loss_type = "ce_qwk"

    # v5: isolated STN identity regularization (stn_identity_reg_weight=0.01) vs v4.
    if experiment == "e2_fgbf_pim_v5":
        train_cfg.stn_identity_reg_weight = 0.01

    # v6: combines stabilized LR, BoundaryAwareLoss (CE + SoftQWK + L1 grade dist), and STN reg.
    if experiment == "e2_fgbf_pim_v6":
        train_cfg.loss_type = "boundary_aware"
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
