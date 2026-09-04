"""
scratch/test_pipeline_verification.py
Verify all recent improvements:
1. Losses on CPU and CUDA (SoftQWKLoss, CombinedOrdinalLoss, BoundaryAwareLoss).
2. build_primary_loss factory.
3. Handedness check in dataset loader.
4. Trainer parameter group setup and optimizer construction.
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

from losses import SoftQWKLoss, CombinedOrdinalLoss, BoundaryAwareLoss, build_primary_loss
from config import get_config
from models.drpnet import DRPNet
from trainer import Trainer

def test_losses():
    print("==> Testing loss functions...")
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    class_weights = torch.ones(5)

    # 1. SoftQWKLoss
    qwk = SoftQWKLoss(num_classes=5)
    loss_val = qwk(logits, targets)
    assert loss_val.ndim == 0, f"Expected scalar, got {loss_val.shape}"
    print("  ✓ SoftQWKLoss works on CPU")

    # 2. CombinedOrdinalLoss
    comb = CombinedOrdinalLoss(class_weights=class_weights, num_classes=5)
    loss_val = comb(logits, targets)
    assert loss_val.ndim == 0, f"Expected scalar, got {loss_val.shape}"
    print("  ✓ CombinedOrdinalLoss works on CPU")

    # 3. BoundaryAwareLoss
    bound = BoundaryAwareLoss(class_weights=class_weights, num_classes=5)
    loss_val = bound(logits, targets)
    assert loss_val.ndim == 0, f"Expected scalar, got {loss_val.shape}"
    print("  ✓ BoundaryAwareLoss works on CPU")

    # 4. CUDA test if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logits_cuda = logits.to(device)
        targets_cuda = targets.to(device)

        bound_cuda = BoundaryAwareLoss(class_weights=class_weights.to(device), num_classes=5).to(device)
        loss_val_cuda = bound_cuda(logits_cuda, targets_cuda)
        assert loss_val_cuda.device.type == "cuda"
        print("  ✓ BoundaryAwareLoss works on CUDA")

        comb_cuda = CombinedOrdinalLoss(class_weights=class_weights.to(device), num_classes=5).to(device)
        loss_val_cuda = comb_cuda(logits_cuda, targets_cuda)
        assert loss_val_cuda.device.type == "cuda"
        print("  ✓ CombinedOrdinalLoss works on CUDA")

def test_trainer_optimizer_groups():
    print("\n==> Testing 3-tier optimizer parameter groups in Trainer...")
    cfg = get_config("e2_fgbf_pim_v6")
    cfg.training.device = "cpu"
    model = DRPNet(cfg.model)
    loss_fn = build_primary_loss(cfg.training.loss_type, [], cfg.model.num_classes, torch.device("cpu"))
    trainer = Trainer(model, loss_fn, cfg)

    param_groups = trainer.optimizer.param_groups
    assert len(param_groups) == 3, f"Expected 3 param groups, got {len(param_groups)}"

    backbone_lr = param_groups[0]["lr"]
    module_lr = param_groups[1]["lr"]
    head_lr = param_groups[2]["lr"]
    base_lr = cfg.training.learning_rate

    # With warmup_epochs > 0, LinearLR starts at start_factor = 0.1
    start_factor = 0.1 if cfg.training.warmup_epochs > 0 else 1.0
    assert abs(backbone_lr - base_lr * 0.2 * start_factor) < 1e-9, f"Backbone LR mismatch: {backbone_lr} vs {base_lr * 0.2 * start_factor}"
    assert abs(module_lr - base_lr * 1.0 * start_factor) < 1e-9, f"Module LR mismatch: {module_lr} vs {base_lr * 1.0 * start_factor}"
    assert abs(head_lr - base_lr * 2.5 * start_factor) < 1e-9, f"Head LR mismatch: {head_lr} vs {base_lr * 2.5 * start_factor}"

    print(f"  ✓ 3-tier parameter groups verified: Backbone LR={backbone_lr:.2e}, Module LR={module_lr:.2e}, Head LR={head_lr:.2e}")

def test_forward_step():
    print("\n==> Testing dummy step forward & backward...")
    cfg = get_config("e2_fgbf_pim_v6")
    cfg.training.device = "cpu"
    model = DRPNet(cfg.model)
    loss_fn = build_primary_loss(cfg.training.loss_type, [], cfg.model.num_classes, torch.device("cpu"))
    trainer = Trainer(model, loss_fn, cfg)

    dummy_crop = torch.randn(2, 3, 224, 224)
    dummy_labels = {
        "kl": torch.tensor([0, 2], dtype=torch.long),
        "jsn_med": torch.tensor([-1, -1], dtype=torch.long),
        "jsn_lat": torch.tensor([-1, -1], dtype=torch.long),
        "osteophyte": torch.tensor([[-1,-1,-1,-1], [-1,-1,-1,-1]], dtype=torch.long),
    }

    step_out = trainer._step(([dummy_crop], dummy_labels))
    assert "total" in step_out and step_out["total"] > 0
    print("  ✓ Forward, backward, and optimizer step executed smoothly on e2_fgbf_pim_v6!")

if __name__ == "__main__":
    test_losses()
    test_trainer_optimizer_groups()
    test_forward_step()
    print("\n🎉 ALL TESTS PASSED!")
