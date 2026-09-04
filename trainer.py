"""
trainer.py
==========
Trainer — encapsulates the full training loop for all CoRD-Net experiments.

Changes from previous version
------------------------------
* Accumulates per-epoch training history (losses, accuracy, macro-F1,
  QWK, MAE, learning rate).
* Collects full logit tensors over val and (optionally) train sets.
* Calls reporting.generate_all_reports() after fit() completes.
* Adds collect_logits() for running a loader through the model without
  updating weights (used by train.py to get train-set metrics at the end).

Architecture, optimizer, scheduler, losses, and checkpoint logic are
unchanged.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from collections import defaultdict
import numpy as np

from config import Config
from losses import MultiTaskLoss
from metrics import evaluate, compute_all_metrics, get_predictions, _to_numpy
from utils import (
    count_parameters, count_all_parameters,
    log_parameter_summary, log_model_summary,
    save_checkpoint, load_checkpoint,
)

logger = logging.getLogger(__name__)


class WarmupPlateauScheduler:
    """Combines LinearLR warmup with ReduceLROnPlateau."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        start_factor: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.plateau_scheduler = plateau_scheduler
        self.warmup_scheduler = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=start_factor, total_iters=warmup_epochs
            )
            if warmup_epochs > 0
            else None
        )
        self.current_epoch = 0

    def step(self, metric: Optional[float] = None) -> None:
        self.current_epoch += 1
        if self.warmup_scheduler is not None and self.current_epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
        elif self.plateau_scheduler is not None:
            if metric is not None:
                self.plateau_scheduler.step(metric)
            else:
                self.plateau_scheduler.step(0.0)

    def state_dict(self) -> Dict:
        return {
            "current_epoch": self.current_epoch,
            "warmup_state": self.warmup_scheduler.state_dict() if self.warmup_scheduler else None,
            "plateau_state": self.plateau_scheduler.state_dict() if self.plateau_scheduler else None,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.current_epoch = state_dict.get("current_epoch", 0)
        if self.warmup_scheduler and state_dict.get("warmup_state"):
            self.warmup_scheduler.load_state_dict(state_dict["warmup_state"])
        if self.plateau_scheduler and state_dict.get("plateau_state"):
            self.plateau_scheduler.load_state_dict(state_dict["plateau_state"])


class Trainer:
    """
    Generic training loop for CoRD-Net experiments (E1–E8).

    Parameters
    ----------
    model:
        DRPNet instance.
    loss_fn:
        MultiTaskLoss for E8, nn.CrossEntropyLoss for E1–E7.
    cfg:
        Top-level Config (model + training settings bundled).
    """

    def __init__(
        self,
        model:   nn.Module,
        loss_fn: nn.Module,
        cfg:     Config,
    ) -> None:
        self.model   = model
        print("\n========== CLASSIFIER INITIALIZATION ==========")

        for name, p in self.model.named_parameters():
            if "classifier" in name:
                print(name)
                print("mean:", p.mean().item())
                print("std :", p.std().item())
            if "classifier.bias" in name:
                print("Classifier bias:", p.detach().cpu())

        print("===============================================\n")
        self.loss_fn = loss_fn
        self.cfg     = cfg
        self.tcfg    = cfg.training
        self.device  = torch.device(
            self.tcfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler    = GradScaler() if self.tcfg.amp else None

        self.epoch      = 0
        self.best_qwk   = -1.0
        self.best_score = -1.0

        # ── Per-epoch history (populated during fit) ──────────────────────
        self.history: Dict[str, List] = {
            "epoch":          [],
            "train_loss":     [],
            "val_loss":       [],
            "train_accuracy": [],
            "val_accuracy":   [],
            "val_macro_f1":   [],
            "val_kl1_recall": [],
            "val_kl1_f1":     [],
            "val_kl2_f1":     [],
            "val_low_grade_min_recall": [],
            "val_score":      [],
            "val_qwk":        [],
            "val_mae":        [],
            "learning_rate":  [],
        }

        log_model_summary(self.model, cfg.experiment)
        log_parameter_summary(self.model, cfg.experiment)

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.tcfg.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.tcfg.learning_rate,
                weight_decay=self.tcfg.weight_decay,
            )
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.tcfg.learning_rate,
            )
        raise ValueError(
            f"Unknown optimizer '{self.tcfg.optimizer}'. Choose: adamw | adam"
        )

    def _build_scheduler(self):
        name = self.tcfg.scheduler.lower()
        warmup_epochs = max(0, getattr(self.tcfg, "warmup_epochs", 0))

        if name == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5
            )
            return WarmupPlateauScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                plateau_scheduler=plateau,
                start_factor=0.1,
            )

        if name == "cosine":
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_epochs
                )
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(1, self.tcfg.epochs - warmup_epochs)
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs]
                )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.tcfg.epochs
            )

        if name == "step":
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_epochs
                )
                main_sched = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=30, gamma=0.1
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, main_sched], milestones=[warmup_epochs]
                )
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

        if name == "none":
            if warmup_epochs > 0:
                return torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_epochs
                )
            return None

        raise ValueError(
            f"Unknown scheduler '{self.tcfg.scheduler}'. Choose: cosine | step | plateau | none"
        )

    # ── Batch unpacking ───────────────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(batch):
        """
        Accept both 1-crop and 3-crop batch formats.

        Format A (E1–E3):  ([global],                  labels_dict)
        Format B (E4–E8):  ([global, medial, lateral],  labels_dict)
        """
        crops, labels = batch
        global_crop = batch[0][0]
        labels = batch[1]
        return global_crop, labels

    def _to_device(self, global_crop, labels):
        global_crop = global_crop.to(self.device, non_blocking=True)
        labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
        return global_crop, labels

    # ── Gradient norm helpers ──────────────────────────────────────────────────

    def _fgbf_gradient_norm(self) -> float:
        if not hasattr(self.model, "fgbf") or self.model.fgbf is None:
            return 0.0

        total = 0.0
        count = 0

        for param in self.model.fgbf.parameters():
            if param.grad is not None:
                total += param.grad.detach().norm().item() ** 2
                count += 1

        return total ** 0.5 if count > 0 else 0.0

    def _backbone_gradient_norm(self) -> float:
        backbone = getattr(self.model, "backbone_features", None)
        if backbone is None:
            return 0.0

        total = 0.0
        count = 0

        for param in backbone.parameters():
            if param.grad is not None:
                total += param.grad.detach().norm().item() ** 2
                count += 1

        return total ** 0.5 if count > 0 else 0.0

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(
        self,
        preds:  Dict,
        labels: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Route to MultiTaskLoss (E8) or CrossEntropyLoss (E1–E7).
        Prototype alignment is added whenever sim_logits is present.
        """
        if isinstance(self.loss_fn, MultiTaskLoss):
            loss_dict = self.loss_fn(preds, labels)
        else:
            if "logits" not in preds:
                raise KeyError(
                    "Model output missing 'logits'. "
                    "Check DRPNet.forward() for this experiment."
                )
            ce = self.loss_fn(preds["logits"], labels["kl"])
            loss_dict = {"kl": ce, "main": ce, "total": ce}

        if "sim_logits" in preds:
            w          = self.tcfg.loss_weights.get("proto", 0.3)
            proto_loss = w * F.cross_entropy(preds["sim_logits"], labels["kl"])
            loss_dict["proto"] = proto_loss
            loss_dict["total"] = loss_dict["total"] + proto_loss

        if "fgbf_logits" in preds:
            kl_targets = labels["kl"]
            low_grade_mask = (kl_targets <= 2)
            if low_grade_mask.any():
                fgbf_ce = F.cross_entropy(
                    preds["fgbf_logits"][low_grade_mask],
                    kl_targets[low_grade_mask],
                )
            else:
                fgbf_ce = 0.0 * preds["fgbf_logits"].sum()

            w_fgbf = getattr(self.cfg.model, "fgbf_loss_weight", 0.15)
            loss_dict["fgbf"] = fgbf_ce
            loss_dict["fgbf_loss"] = fgbf_ce
            loss_dict["weighted_fgbf"] = w_fgbf * fgbf_ce
            loss_dict["weighted_fgbf_loss"] = w_fgbf * fgbf_ce
            loss_dict["total"] = loss_dict["total"] + w_fgbf * fgbf_ce

        return loss_dict

    # ── Single training step ──────────────────────────────────────────────────

    def _step(self, batch) -> Dict[str, float]:
        """One forward → loss → backward → optimizer step."""
        global_crop, labels = self._unpack_batch(batch)
        global_crop, labels = \
            self._to_device(global_crop, labels)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.tcfg.amp):
            preds   = self.model(global_crop)
            # ---------- DEBUG COLLECTION ----------
            if hasattr(self.model, "debug_stats"):
                for k, v in self.model.debug_stats.items():
                    self.debug[k].append(v)
            loss_kv = self._compute_loss(preds, labels)
            logits = preds["logits"]
            pred_cls = logits.argmax(dim=1)
            correct = (pred_cls == labels["kl"]).sum().item()
            count = labels["kl"].size(0)
        total = loss_kv["total"]

        if self.scaler:
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            fgbf_grad = self._fgbf_gradient_norm()
            bb_grad = self._backbone_gradient_norm()
            if self.tcfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            fgbf_grad = self._fgbf_gradient_norm()
            bb_grad = self._backbone_gradient_norm()
            if self.tcfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip
                )
            self.optimizer.step()

        # EMA prototype update AFTER backward
        if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
            drp_emb = getattr(self.model, "_last_drp_emb", None)
            if drp_emb is not None:
                self.model.update_prototypes(drp_emb.detach(), labels["kl"])
        out = {k: v.item() for k, v in loss_kv.items()}
        out["fgbf_grad_norm"] = fgbf_grad
        out["backbone_grad_norm"] = bb_grad
        out["correct"] = correct
        out["count"] = count

        out["logits_mean"] = preds["logits"].detach().mean(dim=0).cpu()

        return out

    # ── Epoch loops ───────────────────────────────────────────────────────────
    def train_epoch(self, loader):
        """Run one full training epoch; return averaged losses + accuracy."""
        self.model.train()

        self.debug = defaultdict(list)

        totals = {}
        n = 0

        correct = 0
        count = 0

        # ADD HERE
        logit_sum = torch.zeros(self.cfg.model.num_classes)
        num_batches = 0

        for batch in loader:
            step = self._step(batch)

            correct += step.pop("correct")
            count += step.pop("count")

            # ADD THESE TWO LINES
            logit_sum += step.pop("logits_mean")
            num_batches += 1

            for k, v in step.items():
                totals[k] = totals.get(k, 0.0) + v

            n += 1

        result = {k: v / max(n, 1) for k, v in totals.items()}
        result["accuracy"] = correct / max(count, 1)

        print("\n" + "=" * 60)
        print("TRAIN EPOCH DEBUG")
        print("=" * 60)

        print("Average logits:", logit_sum / num_batches)

        for k, values in self.debug.items():
            print(f"{k:15s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        print("=" * 60 + "\n")

        return result


    @torch.no_grad()
    def val_epoch(self, loader: Iterator) -> Dict[str, float]:
        """
        Full validation pass.
        Returns averaged losses + accuracy / kappa / mae for the
        per-epoch log.  Does NOT store logits — that is done by
        collect_logits() when full metrics are needed.
        """
        self.model.eval()

        totals: Dict[str, float] = {}
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_fgbf_logits: List[torch.Tensor] = []

        n = 0
        pred_hist = torch.zeros(self.cfg.model.num_classes, dtype=torch.long)

        for batch in loader:
            global_crop, labels = \
                self._unpack_batch(batch)
            global_crop, labels = \
                self._to_device(global_crop, labels)

            preds = self.model(global_crop)

            logits = preds["logits"]          # <-- ADD THIS

            losses = self._compute_loss(preds, labels)

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()

            pred = logits.argmax(dim=1).cpu()
            pred_hist += torch.bincount(
                pred,
                minlength=self.cfg.model.num_classes,
            )

            if n == 0:
                print("First batch prediction counts:",
                    torch.bincount(
                        pred,
                        minlength=self.cfg.model.num_classes
                    ))

            if "logits" in preds:
                all_logits.append(logits.cpu())
                all_labels.append(labels["kl"].cpu())
            if "fgbf_logits" in preds:
                all_fgbf_logits.append(preds["fgbf_logits"].cpu())

            n += 1

        result = {k: v / max(n, 1) for k, v in totals.items()}

        if all_logits:
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            fgbf_cat   = torch.cat(all_fgbf_logits, dim=0) if all_fgbf_logits else None

            m = evaluate(
                logits_cat,
                labels_cat,
                self.cfg.model.num_classes,
            )
            result.update(m)

            full = compute_all_metrics(
                logits_cat,
                labels_cat,
                self.cfg.model.num_classes,
                fgbf_logits=fgbf_cat,
            )
            result["macro_f1"] = full["macro_f1"]
            result["kl0_f1"] = full.get("kl0_f1", 0.0)
            result["kl1_recall"] = full.get("kl1_recall", 0.0)
            result["kl1_f1"] = full.get("kl1_f1", 0.0)
            result["kl2_recall"] = full.get("kl2_recall", 0.0)
            result["kl2_f1"] = full.get("kl2_f1", 0.0)
            result["low_grade_min_recall"] = full.get("low_grade_min_recall", 0.0)
            result["low_grade_min_f1"] = full.get("low_grade_min_f1", 0.0)
            result["low_grade_worst_class"] = full.get("low_grade_worst_class", -1)
            if fgbf_cat is not None:
                result.update({k: v for k, v in full.items() if k.startswith("fgbf_")})

            label_hist = torch.bincount(
                labels_cat,
                minlength=self.cfg.model.num_classes,
            )

            print("\n========== VALIDATION HISTOGRAM ==========")
            print("Pred :", pred_hist.tolist())
            print("True :", label_hist.tolist())
            print("==========================================\n")

        return result

    # ── Logit collection (used for final reporting) ───────────────────────────

    @torch.no_grad()
    def collect_logits(
        self, loader: Iterator
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

        self.model.eval()

        all_logits = []
        all_labels = []
        all_fgbf_logits = []

        for batch in loader:
            global_crop, labels = self._unpack_batch(batch)

            global_crop = global_crop.to(
                self.device,
                non_blocking=True,
            )

            preds = self.model(global_crop)

            if "logits" in preds:
                all_logits.append(preds["logits"].cpu())
                all_labels.append(labels["kl"])
            if "fgbf_logits" in preds:
                all_fgbf_logits.append(preds["fgbf_logits"].cpu())

        if not all_logits:
            empty = np.zeros((0, self.cfg.model.num_classes), dtype=np.float32)
            return empty, np.zeros(0, dtype=np.int64), None

        fgbf_np = _to_numpy(torch.cat(all_fgbf_logits, dim=0)) if all_fgbf_logits else None

        return (
            _to_numpy(torch.cat(all_logits, dim=0)),
            _to_numpy(torch.cat(all_labels, dim=0)).astype(int),
            fgbf_np,
        )

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(
        self,
        epoch:        int,
        train_losses: Dict,
        val_losses:   Optional[Dict] = None,
        tag:          str = "latest",
    ) -> None:
        state = {
            "experiment":           self.cfg.experiment,
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
                                    if self.scheduler else None,
            "best_score":           self.best_score,
            "best_qwk":             self.best_qwk,
            "train_losses":         train_losses,
            "val_losses":           val_losses or {},
            "history":              self.history,
        }
        save_checkpoint(
            state,
            self.tcfg.checkpoint_dir,
            filename=f"{self.cfg.experiment}_{tag}.pt",
        )

    def resume(self, checkpoint_path: str | Path) -> None:
        """Restore model, optimizer, scheduler, epoch, and history."""
        ckpt = load_checkpoint(
            checkpoint_path,
            self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            device    = self.device,
        )
        self.epoch      = ckpt.get("epoch", 0)
        self.best_score = ckpt.get("best_score", -1.0)
        self.best_qwk   = ckpt.get("best_qwk", -1.0)
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info(
            "Resumed from epoch %d  (best_score=%.4f, best_qwk=%.4f)",
            self.epoch, self.best_score, self.best_qwk,
        )

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader:  Iterator,
        val_loader:    Optional[Iterator] = None,
        test_loader:   Optional[Iterator] = None,
        resume:        Optional[str | Path] = None,
        results_dir:   str = "results",
    ) -> None:
        """
        Run the full training loop, then generate all reports.

        Parameters
        ----------
        train_loader:  DataLoader for the training split.
        val_loader:    DataLoader for the validation split.
        test_loader:   DataLoader for the test split (used in final reporting).
        resume:        Path to checkpoint to resume from.
        results_dir:   Root results directory for reporting output.
        """
        if resume is not None:
            self.resume(resume)

        start_epoch = self.epoch + 1
        logger.info(
            "Training %s | epochs %d→%d | device=%s",
            self.cfg.experiment.upper(),
            start_epoch,
            self.tcfg.epochs,
            self.device,
        )

        train_losses: Dict[str, float] = {}
        patience_counter = 0

        for epoch in range(start_epoch, self.tcfg.epochs + 1):
            self.epoch = epoch
            t0 = time.time()

            train_losses = self.train_epoch(train_loader)
            elapsed      = time.time() - t0
            current_lr   = self.optimizer.param_groups[0]["lr"]

            log_parts = [
                f"epoch={epoch}/{self.tcfg.epochs}",
                f"time={elapsed:.1f}s",
                f"lr={current_lr:.2e}",
            ]
            log_parts += [f"train/{k}={v:.4f}" for k, v in train_losses.items()]

            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.val_epoch(val_loader)
                log_parts  += [f"val/{k}={v:.4f}" for k, v in val_metrics.items()]
                val_total   = val_metrics.get("total", float("inf"))

            val_qwk = val_metrics.get("kappa", -1.0)
            val_macro_f1 = val_metrics.get("macro_f1", 0.0)
            val_kl1_f1 = val_metrics.get("kl1_f1", 0.0)
            val_kl2_f1 = val_metrics.get("kl2_f1", 0.0)
            val_low_grade_min_recall = val_metrics.get("low_grade_min_recall", 0.0)

            # checkpoint_monitor dispatch. NOTE: this field existed in config.py
            # before this patch but was never read here — every experiment to
            # date (including e2_fgbf_pim_v2) actually ran on "kl1_only" below,
            # regardless of what checkpoint_monitor said. That default is fixed
            # in config.py alongside this change so the two stay honest.
            #
            #   "qwk"      — pure QWK selection (kept for a true ablation only;
            #                not used by any experiment run so far).
            #   "kl1_only" — 0.5*macro_f1 + 0.5*kl1_f1, no regression guard.
            #                This is what e2_fgbf_pim_v2 actually ran on, and
            #                is kept as-is for exact reproducibility.
            #   "score"    — composite covering both low-grade classes in
            #                tension (KL1 and KL2), gated by a regression guard
            #                on whichever of KL0/KL1/KL2 is currently weakest,
            #                so improving one can't silently collapse another.
            monitor = getattr(self.tcfg, "checkpoint_monitor", "kl1_only")

            if monitor == "score":
                score = 0.4 * val_macro_f1 + 0.3 * val_kl1_f1 + 0.3 * val_kl2_f1
                min_epochs = getattr(self.tcfg, "min_epochs_before_early_stop", 15)
                floor = getattr(self.tcfg, "low_grade_recall_floor", 0.30)
                # Exempt the warmup window: metrics are noisy before the model
                # stabilizes, and this exemption guarantees at least one
                # checkpoint gets saved even if the floor is never cleared
                # again later in the run.
                guard_ok = (val_low_grade_min_recall >= floor) or (epoch < min_epochs)
            elif monitor == "kl1_only":
                score = 0.5 * val_macro_f1 + 0.5 * val_kl1_f1
                guard_ok = True
            else:  # "qwk"
                score = val_qwk
                guard_ok = True

            log_parts.append(f"val/score={score:.4f}")

            if score > self.best_score and guard_ok:
                self.best_score = score
                self.best_qwk = val_qwk
                patience_counter = 0
                self._save(epoch, train_losses, val_metrics, tag="best")
            else:
                patience_counter += 1
                if score > self.best_score and not guard_ok:
                    logger.warning(
                        "Epoch %d: score %.4f would be a new best, but low-grade "
                        "min recall %.3f (worst class idx %d) is below floor %.2f "
                        "— not saving as 'best' this epoch.",
                        epoch, score, val_low_grade_min_recall,
                        int(val_metrics.get("low_grade_worst_class", -1)), floor,
                    )

            logger.info(" | ".join(log_parts))

            # ── Record history ──────────────────────────────────────────
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_losses.get("total", float("nan")))
            self.history["val_loss"].append(val_metrics.get("total",    float("nan")))
            self.history["train_accuracy"].append(train_losses.get("accuracy", float("nan")))
            self.history["val_accuracy"].append(val_metrics.get("accuracy",  float("nan")))
            self.history["val_macro_f1"].append( val_metrics.get("macro_f1", float("nan")))
            self.history["val_kl1_recall"].append(val_metrics.get("kl1_recall", float("nan")))
            self.history["val_kl1_f1"].append(val_metrics.get("kl1_f1", float("nan")))
            self.history["val_kl2_f1"].append(val_metrics.get("kl2_f1", float("nan")))
            self.history["val_low_grade_min_recall"].append(val_metrics.get("low_grade_min_recall", float("nan")))
            self.history["val_score"].append(score)
            self.history["val_qwk"].append(      val_metrics.get("kappa",    float("nan")))
            self.history["val_mae"].append(      val_metrics.get("mae",      float("nan")))
            self.history["learning_rate"].append(current_lr)

            if self.scheduler is not None:
                if isinstance(self.scheduler, WarmupPlateauScheduler):
                    self.scheduler.step(metric=score)
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            if epoch % self.tcfg.save_every == 0:
                self._save(epoch, train_losses, tag=f"epoch{epoch:04d}")

            min_epochs = getattr(self.tcfg, "min_epochs_before_early_stop", 15)
            if self.tcfg.patience is not None and epoch >= min_epochs and patience_counter >= self.tcfg.patience:
                logger.info(
                    "Early stopping triggered at epoch %d (no composite score improvement for %d epochs; min_epochs=%d).",
                    epoch, patience_counter, min_epochs
                )
                print(f"Early stopping at epoch {epoch}")
                break

        # Final checkpoint
        self._save(self.epoch, train_losses, tag="final")
        
        best_ckpt_path = Path(self.tcfg.checkpoint_dir) / f"{self.cfg.experiment}_best.pt"
        if best_ckpt_path.exists():
            logger.info("Restoring best model checkpoint (best_score=%.4f, val_qwk=%.4f) from %s for evaluation...", self.best_score, self.best_qwk, best_ckpt_path)
            load_checkpoint(best_ckpt_path, self.model, device=self.device)

        logger.info("Training complete. Generating reports …")

        # ── Post-training reporting ─────────────────────────────────────────
        self._run_reporting(
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            results_dir  = results_dir,
        )

    def _run_reporting(
        self,
        train_loader: Optional[Iterator],
        val_loader:   Optional[Iterator],
        test_loader:  Optional[Iterator],
        results_dir:  str,
    ) -> None:
        """Collect final logits and call generate_all_reports()."""
        from reporting import ResultsWriter, generate_all_reports
        from utils import count_parameters

        if val_loader is None:
            logger.warning(
                "No val_loader provided — skipping reporting."
            )
            return

        writer = ResultsWriter(self.cfg.experiment, results_dir)

        logger.info("Collecting val logits …")
        val_logits, val_labels, val_fgbf_logits = self.collect_logits(val_loader)

        train_logits = train_labels = train_fgbf_logits = None
        if train_loader is not None:
            logger.info("Collecting train logits …")
            train_logits, train_labels, train_fgbf_logits = self.collect_logits(train_loader)


        test_logits = test_labels = test_fgbf_logits = None
        if test_loader is not None:
            logger.info("Collecting test logits …")
            test_logits, test_labels, test_fgbf_logits = self.collect_logits(test_loader)
        generate_all_reports(
            writer            = writer,
            history           = self.history,
            train_logits      = train_logits,
            train_labels      = train_labels,
            val_logits        = val_logits,
            val_labels        = val_labels,
            test_logits       = test_logits,
            test_labels       = test_labels,
            num_classes       = self.cfg.model.num_classes,
            parameters        = count_parameters(self.model),
            results_dir       = results_dir,
            val_fgbf_logits   = val_fgbf_logits,
            test_fgbf_logits  = test_fgbf_logits,
            train_fgbf_logits = train_fgbf_logits,
        )

    # ── Synthetic stub (run_experiment.py only) ───────────────────────────────

    def stub_fit(self, steps: int = 3, image_size: int = 112) -> Dict[str, float]:
        """
        Run *steps* synthetic mini-batch passes for integration testing.
        Not used in real training.
        """
        from utils import make_labels_stub

        B         = self.tcfg.batch_size
        K         = self.cfg.model.num_classes

        self.model.train()
        last_losses: Dict[str, float] = {}

        for step in range(1, steps + 1):
            self.optimizer.zero_grad(set_to_none=True)

            g  = torch.randn(B, 3, image_size, image_size, device=self.device)
            labels = make_labels_stub(B, K, self.device)
            preds     = self.model(g)
            loss_dict = self._compute_loss(preds, labels)
            loss_dict["total"].backward()

            if self.tcfg.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.gradient_clip
                )
            self.optimizer.step()

            if self.cfg.model.use_pgr and hasattr(self.model, "update_prototypes"):
                drp_emb = getattr(self.model, "_last_drp_emb", None)
                if drp_emb is not None:
                    self.model.update_prototypes(drp_emb.detach(), labels["kl"])

            last_losses = {k: v.item() for k, v in loss_dict.items()}
            parts = " | ".join(f"{k}={v:.4f}" for k, v in last_losses.items())
            logger.info("  step %d/%d  %s", step, steps, parts)

        return last_losses
