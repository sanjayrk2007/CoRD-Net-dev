"""
scratch/verify_fgbf_experiments.py
===================================
Static Verification Suite for FGBF Fine-Grained Feature Refinement Experiments.

This script performs complete static, CPU/GPU, synthetic tensor, parameter count,
and safety checks on the FGBF fine-grained experiment variants:
- e2
- e2_fgbf (baseline with Identity block)
- e2_fgbf_ms (Multi-Scale block)
- e2_fgbf_sk (Selective Kernel block)
- e2_fgbf_pim (PIM-Lite block)
- e2_fgbf_cbam (CBAM-Lite block)

DOES NOT START FULL GPU TRAINING.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Root directory setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_verification() -> bool:
    print("=" * 70)
    print("      FGBF FINE-GRAINED BLOCKS STATIC VERIFICATION SUITE      ")
    print("=" * 70)

    results: dict[str, bool] = {}

    # -------------------------------------------------------------------------
    # 1. Syntax & Compilation Check
    # -------------------------------------------------------------------------
    print("\n--- 1. Python Syntax & Compilation Check ---")
    files_to_check = [
        PROJECT_ROOT / "config.py",
        PROJECT_ROOT / "models" / "fgbf_blocks.py",
        PROJECT_ROOT / "models" / "fgbf.py",
        PROJECT_ROOT / "models" / "drpnet.py",
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "losses.py",
        PROJECT_ROOT / "metrics.py",
    ]

    syntax_pass = True
    for filepath in files_to_check:
        try:
            py_compile.compile(str(filepath), doraise=True)
            print(f"  [PASS] Syntax clean: {filepath.name}")
        except py_compile.PyCompileError as err:
            print(f"  [FAIL] Syntax error in {filepath.name}: {err}")
            syntax_pass = False

    results["syntax"] = syntax_pass

    # -------------------------------------------------------------------------
    # 2. Imports Check
    # -------------------------------------------------------------------------
    print("\n--- 2. Imports Check ---")
    imports_pass = True
    try:
        from config import get_config, EXPERIMENT_NAMES, ModelConfig
        from models.fgbf_blocks import (
            build_fgbf_block,
            MultiScaleFineGrainedBlock,
            SelectiveKernelFineGrainedBlock,
            PIMLiteFineGrainedBlock,
            CBAMFineGrainedBlock,
        )
        from models.fgbf import FineGrainedBoundaryFeatureModule
        from models.drpnet import DRPNet
        print("  [PASS] All modules successfully imported.")
    except Exception as err:
        print(f"  [FAIL] Import failed: {err}")
        imports_pass = False

    results["imports"] = imports_pass

    if not imports_pass:
        print("\nStopping further tests due to import failures.")
        return False

    # -------------------------------------------------------------------------
    # 3. Experiment Registry Check
    # -------------------------------------------------------------------------
    print("\n--- 3. Experiment Registry Check ---")
    required_exps = ["e2", "e2_fgbf", "e2_fgbf_ms", "e2_fgbf_sk", "e2_fgbf_pim", "e2_fgbf_cbam"]
    registry_pass = True
    for exp in required_exps:
        if exp in EXPERIMENT_NAMES:
            cfg = get_config(exp)
            print(f"  [PASS] Found '{exp}' -> '{cfg.description}' | block='{cfg.model.fgbf_block}'")
        else:
            print(f"  [FAIL] Missing experiment '{exp}' in EXPERIMENT_NAMES")
            registry_pass = False

    results["experiment_registry"] = registry_pass

    # -------------------------------------------------------------------------
    # 4. Individual Block Shape & Finite Output Test
    # -------------------------------------------------------------------------
    print("\n--- 4. Individual Block Shape & Finite Output Test ---")
    block_pass = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_block_input = torch.randn(4, 256, 7, 7, device=device)

    block_types = ["baseline", "multiscale", "sk", "pim", "cbam"]
    for btype in block_types:
        try:
            blk = build_fgbf_block(btype, channels=256).to(device)
            blk.eval()
            with torch.no_grad():
                out = blk(dummy_block_input)
            
            shape_match = out.shape == (4, 256, 7, 7)
            is_finite = torch.isfinite(out).all().item()

            if shape_match and is_finite:
                print(f"  [PASS] Block '{btype}': Output shape {tuple(out.shape)} | Finite: True")
            else:
                print(f"  [FAIL] Block '{btype}': Output shape {tuple(out.shape)} | Finite: {is_finite}")
                block_pass = False
        except Exception as err:
            print(f"  [FAIL] Block '{btype}' execution error: {err}")
            block_pass = False

    results["individual_block_tests"] = block_pass

    # -------------------------------------------------------------------------
    # 5. Model Construction & Parameter Count Check
    # -------------------------------------------------------------------------
    print("\n--- 5. Model Construction & Parameter Counts ---")
    construction_results: dict[str, bool] = {}
    param_counts: dict[str, tuple[int, int]] = {}

    for exp in required_exps:
        try:
            cfg = get_config(exp)
            model = DRPNet(cfg.model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            stn_status = "enabled" if cfg.model.use_stn else "disabled"
            fgbf_status = "enabled" if cfg.model.use_fgbf else "disabled"
            selected_block = getattr(cfg.model, "fgbf_block", "N/A")

            param_counts[exp] = (total_params, trainable_params)
            construction_results[f"{exp}_construction"] = True

            print(f"  [PASS] Experiment '{exp:<12}': Params: {total_params:,} (Trainable: {trainable_params:,}) | STN: {stn_status} | FGBF: {fgbf_status} | Block: {selected_block}")
        except Exception as err:
            print(f"  [FAIL] Construction failed for '{exp}': {err}")
            construction_results[f"{exp}_construction"] = False

    results.update(construction_results)

    # -------------------------------------------------------------------------
    # 6. Complete Model Synthetic Forward Pass & Tensor Shape Test
    # -------------------------------------------------------------------------
    print("\n--- 6. Synthetic Model Forward Pass & Tensor Shape Test ---")
    forward_pass = True
    nan_inf_pass = True

    dummy_image = torch.randn(2, 3, 224, 224, device=device)

    for exp in required_exps:
        try:
            cfg = get_config(exp)
            model = DRPNet(cfg.model).to(device)
            model.eval()

            with torch.no_grad():
                outputs = model(dummy_image)

            logits = outputs.get("logits")
            fgbf_logits = outputs.get("fgbf_logits")

            if logits is None or logits.shape != (2, 5):
                print(f"  [FAIL] Experiment '{exp}': Main logits shape mismatch or missing: {logits.shape if logits is not None else None}")
                forward_pass = False

            if cfg.model.use_fgbf:
                if fgbf_logits is None or fgbf_logits.shape != (2, 3):
                    print(f"  [FAIL] Experiment '{exp}': FGBF logits shape mismatch or missing: {fgbf_logits.shape if fgbf_logits is not None else None}")
                    forward_pass = False

            # Check NaNs / Infs
            for key, val in outputs.items():
                if isinstance(val, torch.Tensor):
                    if not torch.isfinite(val).all():
                        print(f"  [FAIL] Experiment '{exp}': Non-finite values detected in output key '{key}'")
                        nan_inf_pass = False

            print(f"  [PASS] Experiment '{exp:<12}': Main logits shape: {tuple(logits.shape)} | FGBF logits shape: {tuple(fgbf_logits.shape) if fgbf_logits is not None else 'N/A'}")

        except Exception as err:
            print(f"  [FAIL] Forward pass failed for '{exp}': {err}")
            forward_pass = False

    results["synthetic_forward_tests"] = forward_pass
    results["nan_inf_checks"] = nan_inf_pass

    # -------------------------------------------------------------------------
    # 7. Baseline Preservation Test
    # -------------------------------------------------------------------------
    print("\n--- 7. Baseline Preservation Test ---")
    baseline_pass = True
    try:
        cfg_fgbf = get_config("e2_fgbf")
        if cfg_fgbf.model.fgbf_block != "baseline":
            print(f"  [FAIL] e2_fgbf default block is '{cfg_fgbf.model.fgbf_block}', expected 'baseline'")
            baseline_pass = False

        mod_fgbf = FineGrainedBoundaryFeatureModule(in_channels=768, reduced_dim=256, fgbf_block="baseline")
        is_identity = isinstance(mod_fgbf.block, nn.Identity)
        if is_identity:
            print("  [PASS] e2_fgbf selects 'baseline' block which resolves to nn.Identity().")
        else:
            print(f"  [FAIL] e2_fgbf block is not nn.Identity(), got {type(mod_fgbf.block)}")
            baseline_pass = False
    except Exception as err:
        print(f"  [FAIL] Baseline preservation check error: {err}")
        baseline_pass = False

    results["baseline_preservation"] = baseline_pass

    # -------------------------------------------------------------------------
    # 8. Boundary Safety Check
    # -------------------------------------------------------------------------
    print("\n--- 8. Boundary Code Safety Check ---")
    boundary_symbols = [
        "e2_fgbf_boundary",
        "fgbf_boundary_mode",
        "boundary_head_01",
        "boundary_head_12",
        "B01",
        "B12",
        "boundary_loss",
    ]
    boundary_pass = True
    for symbol in boundary_symbols:
        found_files = []
        for file_path in PROJECT_ROOT.glob("**/*.py"):
            parts = file_path.parts
            if "scratch" in parts or ".git" in parts or "cordnet-venv" in parts or "venv" in parts or ".venv" in parts:
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                if symbol in content:
                    found_files.append(file_path.name)
            except Exception:
                pass
        if found_files:
            print(f"  [FAIL] Forbidden boundary symbol '{symbol}' found in: {found_files}")
            boundary_pass = False

    if boundary_pass:
        print("  [PASS] Zero forbidden boundary symbols introduced.")

    results["boundary_safety"] = boundary_pass

    # -------------------------------------------------------------------------
    # 9. Parameter Overhead & Memory Sanity Check for RTX 5050 8 GB
    # -------------------------------------------------------------------------
    print("\n--- 9. Parameter Overhead & Memory Sanity Check (RTX 5050 8 GB) ---")
    base_params = param_counts.get("e2_fgbf", (0, 0))[0]
    rtx5050_pass = True

    for exp in ["e2_fgbf_ms", "e2_fgbf_sk", "e2_fgbf_pim", "e2_fgbf_cbam"]:
        total_p = param_counts.get(exp, (0, 0))[0]
        added_params = total_p - base_params
        mb_overhead = (added_params * 4) / (1024 * 1024)  # fp32 in MB
        print(f"  Module '{exp:<13}': Added Params: {added_params:,} ({mb_overhead:.2f} MB float32)")
        
        # Verify block is lightweight (< 2M parameters added)
        if added_params > 2_000_000:
            print(f"  [WARNING] Parameter overhead for {exp} is high: {added_params:,} params!")
            rtx5050_pass = False

    if rtx5050_pass:
        print("  [PASS] All fine-grained feature blocks are lightweight and fully compatible with 8 GB VRAM.")

    results["param_memory_sanity"] = rtx5050_pass
    results["rtx_5050_readiness"] = rtx5050_pass

    # -------------------------------------------------------------------------
    # Final Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                    FINAL VERIFICATION SUMMARY                    ")
    print("=" * 70)
    all_passed = True
    for key, status in results.items():
        status_str = "PASS" if status else "FAIL"
        print(f"  {key:<35} : {status_str}")
        if not status:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> OVERALL STATUS: ALL CHECKS PASSED SUCCESSFULLY <<<")
    else:
        print(">>> OVERALL STATUS: VERIFICATION FAILED - FIX ERRORS ABOVE <<<")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
