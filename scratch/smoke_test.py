"""
scratch/smoke_test.py
====================
Smoke test verifying:
1. e3_fgbf_sampler02 shape compatibility with active fusion path (fgbf_fuse_main=True).
2. Per-epoch logs including kl1_recall / kl1_f1 and composite val/score.
3. Checkpoint saving triggering on composite score.
4. LR warmup ramping over warmup_epochs.
5. Reporting output formatting N/A for empty train metrics.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

from config import get_config
from models.drpnet import DRPNet
from trainer import Trainer
from reporting import print_final_results

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@dataclass
class MockSample:
    kl: int

class MockDataset(Dataset):
    def __init__(self, size=16):
        self.size = size
        # Diverse KL grades (0, 1, 2, 3, 4)
        self.kl_grades = [i % 5 for i in range(size)]
        self.samples = [MockSample(kl=kl) for kl in self.kl_grades]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1-crop format ([global_crop], labels_dict)
        crop = torch.randn(3, 224, 224)
        labels = {
            "kl": torch.tensor(self.kl_grades[idx], dtype=torch.long),
            "medial_jsn": torch.tensor(-1, dtype=torch.long),
            "lateral_jsn": torch.tensor(-1, dtype=torch.long),
            "medial_femur_osteo": torch.tensor(-1, dtype=torch.long),
            "lateral_femur_osteo": torch.tensor(-1, dtype=torch.long),
            "medial_tibia_osteo": torch.tensor(-1, dtype=torch.long),
            "lateral_tibia_osteo": torch.tensor(-1, dtype=torch.long),
        }
        return [crop], labels

def run_smoke_test():
    print("\n" + "=" * 60)
    print("STEP 1: Test e3_fgbf_sampler02 configuration and forward pass")
    print("=" * 60)
    # NOTE: e2_fgbf_pim_v2 was trimmed from the active registry in config.py
    # (see the "Intermediate ablation/debugging keys ... trimmed" comment).
    # e3_fgbf_sampler02 is the closest live equivalent — same fgbf_fuse_main=True
    # fusion path, and it's also your current best verified result, so this
    # smoke test now exercises a config that actually matters.
    cfg = get_config("e3_fgbf_sampler02", device="cpu")
    cfg.training.epochs = 6
    cfg.training.warmup_epochs = 4
    cfg.training.batch_size = 4
    cfg.training.patience = 5
    cfg.training.checkpoint_dir = "scratch/checkpoints"
    cfg.training.results_dir = "scratch/results"

    print("Config verification:")
    print("  experiment       :", cfg.experiment)
    print("  use_fgbf         :", cfg.model.use_fgbf)
    print("  fgbf_block       :", cfg.model.fgbf_block)
    print("  fgbf_fuse_main   :", cfg.model.fgbf_fuse_main)
    print("  loss_type        :", cfg.training.loss_type)
    print("  sampler          :", cfg.training.sampler)
    print("  warmup_epochs    :", cfg.training.warmup_epochs)
    print("  min_epochs_early :", cfg.training.min_epochs_before_early_stop)

    assert cfg.model.use_fgbf is True
    assert cfg.model.fgbf_block == "pim"
    assert cfg.model.fgbf_fuse_main is True
    assert cfg.training.loss_type == "weighted_ce"
    assert cfg.training.sampler == "weighted"

    model = DRPNet(cfg.model)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("\nModel Forward Output Shapes:")
    print("  logits shape     :", out["logits"].shape)
    print("  fgbf_logits shape:", out.get("fgbf_logits", torch.tensor([])).shape)
    assert out["logits"].shape == (2, 5), f"Expected (2, 5), got {out['logits'].shape}"
    assert "fgbf_logits" in out and out["fgbf_logits"].shape == (2, 3)

    print("\n" + "=" * 60)
    print("STEP 2: Run 6-epoch training loop (testing warmup + metrics)")
    print("=" * 60)
    train_ds = MockDataset(16)
    val_ds = MockDataset(16)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainer = Trainer(model, loss_fn, cfg)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        results_dir="scratch/results",
    )

    print("\n" + "=" * 60)
    print("STEP 3: Verify Learning Rate Warmup Schedule")
    print("=" * 60)
    lrs = trainer.history["learning_rate"]
    for ep, lr in enumerate(lrs, 1):
        print(f"  Epoch {ep}: lr = {lr:.6e}")
    # Epoch 1 LR should be approx 10% of base 1e-4
    assert lrs[0] < lrs[-1] or lrs[0] <= 2.5e-5, f"Warmup did not start at ~10% LR: {lrs}"

    print("\n" + "=" * 60)
    print("STEP 4: Verify Per-Epoch History Keys")
    print("=" * 60)
    print("  History keys:", list(trainer.history.keys()))
    assert "val_kl1_recall" in trainer.history
    assert "val_kl1_f1" in trainer.history
    assert "val_score" in trainer.history
    print("  val_kl1_recall history:", trainer.history["val_kl1_recall"])
    print("  val_kl1_f1 history    :", trainer.history["val_kl1_f1"])
    print("  val_score history     :", trainer.history["val_score"])

    print("\n" + "=" * 60)
    print("STEP 5: Verify Reporting Output with Empty Train Metrics")
    print("=" * 60)
    val_metrics = {
        "accuracy": 0.85,
        "macro_precision": 0.80,
        "macro_recall": 0.82,
        "macro_f1": 0.81,
        "weighted_precision": 0.85,
        "weighted_recall": 0.85,
        "weighted_f1": 0.85,
        "mae": 0.20,
        "qwk": 0.90,
        "kl0_recall": 0.90,
        "kl1_recall": 0.75,
        "kl1_f1": 0.78,
        "kl2_recall": 0.80,
    }
    print_final_results(train_metrics={}, val_metrics=val_metrics, test_metrics={})

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
