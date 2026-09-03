# CoRD-Net Experiment Guide

## Prerequisites

-   Activate the virtual environment:

    ``` bash
    source ~/venvs/ml/bin/activate
    ```

-   Install all dependencies.

-   Configure the OAI dataset paths.

-   Verify CUDA availability.

## Experiment Pipeline

  Experiment   Modules Enabled
  ------------ --------------------------
  E1           Baseline ConvNeXt
  E2           STN
  E3           STN + DRP
  E4           STN + Compartment Branch
  E5           E4 + DRP
  E6           E5 + PGR
  E7           E6 + RTC
  E8           E7 + Auxiliary Heads

## Running

``` bash
source ~/venvs/ml/bin/activate
python train.py --config configs/eX.yaml
```

Examples:

``` bash
python train.py --config configs/e1.yaml
python train.py --config configs/e2.yaml
python train.py --config configs/e4.yaml
```

## Workflow

1.  Select the configuration.
2.  Verify enabled modules.
3.  Start training.
4.  Monitor train/validation metrics.
5.  Save checkpoints.
6.  Compare with the previous experiment.

## Metrics

-   Train Accuracy
-   Validation Accuracy
-   Test Accuracy
-   Macro Precision
-   Macro Recall
-   Macro F1
-   QWK
-   MAE
-   Confusion Matrix

## Debugging

-   Visualize STN outputs.
-   Visualize compartment crops.
-   Visualize DRP attention maps.
-   Remove visualization code before long training runs.

## FGBF Module (Fine-Grained Boundary Feature)

The FGBF module targets the chronic misclassification of KL1 ("doubtful" OA) as KL0 or KL2 by extracting high-frequency boundary and joint-space features.

Key parameters in `ModelConfig`:
- **`use_fgbf`** (`bool`): Enables the FGBF branch (`models/fgbf.py`).
- **`fgbf_block`** (`str`): Selects the boundary feature extraction block (`"baseline"`, `"multiscale"`, `"sk"`, `"pim"`, `"cbam"`). Validated standard is `"pim"` (PIM-Lite).
- **`fgbf_fuse_main`** (`bool`): When `True`, concatenates the 256-d boundary feature directly into the primary 5-way classifier input representation (`parts` list in `models/drpnet.py`), directly guiding KL grade prediction rather than only acting via auxiliary loss.
- **`fgbf_loss_weight`** (`float`): Loss multiplier for the auxiliary 3-way (KL0, KL1, KL2) boundary classification loss in `trainer.py` (default: 0.15).

## Current Status

-   **E1**: Baseline ConvNeXt complete (results in `results/e1`).
-   **E2**: STN localization complete (results in `results/e2`).
-   **E3**: Dual-Intensity Stem complete (results in `results/e3`).
-   **E4**: STN + Compartment Branches complete (results in `results/e4`).
-   **E5**: STN + Compartments + DRP complete (results in `results/e5`).
-   **E2 FGBF Ablations**: `e2_fgbf`, `e2_fgbf_ms`, `e2_fgbf_sk`, `e2_fgbf_cbam`, `e2_fgbf_pim` complete; `e2_fgbf_pim_v2` introduces active main-classifier fusion (`fgbf_fuse_main=True`).
-   **E6--E8**: Pending execution with fused FGBF.

## Notes

-   Keep random seeds fixed.
-   Archive logs, checkpoints, and metrics.
-   Compare each experiment against the previous one before proceeding.
