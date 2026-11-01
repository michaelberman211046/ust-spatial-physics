# Additional analysis

This directory contains the saved-checkpoint evaluation and controlled analysis code accompanying the ultrasound travel-time reconstruction manuscript. The archived original software release remains unchanged.

## Contents

- `aggregate_evaluation`: complete held-out learned-model evaluation, regularized bent-ray comparison on the predefined 100-sample subset, and architecture/configuration extraction.
- `initial_refinement`: the earlier three-sample physical-refinement evaluation and its original configuration, with measured reconstruction and implementation-metadata extraction.
- `reconstruction_analysis`: matched spatial-input removal/retraining, expanded 20-sample Eikonal refinement, RF-picker repeatability and imposed-delay recovery, processed-ToF perturbations, measured-support sensitivity, forward-grid convergence, and fresh physical optimization at multiple reconstruction grids.
- `figures/generate_analysis_figures.py`: extract publication figures from saved numerical outputs, without training or reconstruction.
- `summarize_saved_analyses.py`: collect saved metrics and planned/completed counts into separate summary files.
- `runs`: PowerShell command sequences, pasted from the **project root** (the directory containing `Data` and `additional_analysis`).

Each evaluation contains its private numerical engine to preserve the evaluated configuration. Requirements are the repository's Python environment, including CUDA-enabled PyTorch for the GPU commands, NumPy, SciPy, scikit-fmm, h5py, matplotlib and OpenCV. Inputs include the saved chunked synthetic data, saved train/validation/test split, retained model checkpoints, acquisition MAT file and measured reconstruction/setup outputs. See each script's `--help` and JSON configuration for exact paths and settings.

## Run from a fresh output directory

1. `runs/RUN_AGGREGATE_EVALUATION.txt` reproduces the complete-test and bent-ray comparisons.
2. `runs/RUN_INITIAL_REFINEMENT.txt` reproduces the original small refinement evaluation.
3. `runs/RUN_CONDITIONING_AND_SENSITIVITY.txt` runs the matched ablation and extended analyses. `runs/RUN_SENSITIVITY_ONLY.txt` runs only the extended analyses when a matched ablation is already available.
4. `runs/GENERATE_ANALYSIS_FIGURES.txt` extracts the figures from saved outputs.

The public copy uses neutral package names and fresh result directories. **Keep a running numerical experiment on its original code/configuration.** Source hashes and protocol checks intentionally reject resuming a source-modified run. This publication copy is for new reproducibility runs, not an in-place replacement for code already running elsewhere.

The reconstruction analysis protects its original inputs and writes to its declared new output directory. Configurations freeze the selected samples before their outcomes are examined. Some earlier evaluation tools have less extensive write protection: always use the supplied separate result paths and preserve original checkpoints. Do not point output folders at source data or archived runs.

## Interpret the saved results

The ablation retrains only the spatial correction network, under a common six-epoch budget on 256 training and 64 validation samples, with fixed upstream components and one training seed. It is a targeted architectural comparison, separate from the original full training schedule. Both fitted variants are evaluated on all 2000 test samples. Test-sample bootstrap intervals are conditional on the fitted networks.

The expanded refinement uses 20 predefined synthetic test samples. All selected cases, including rejected/unchanged updates, enter the saved analysis. Known generating fields and the FWI comparison image are used for evaluation after optimization. Support tests hold the measured entering image and the nominal evaluation region fixed. Grid tests distinguish forward numerical convergence, interpolation-only controls and fresh physical reoptimization; they do not retrain a finer-grid neural model.

RF uncertainty outputs distinguish template/noise repeatability and known incremental time-shift recovery from absolute first-arrival accuracy. Unusable pre-arrival noise estimates are recorded by channel; added-noise tests use the explicitly reported eligible subset, while template-only and known-delay tests retain the predetermined channel list. Noise is added on top of the waveform's existing noise.

Long physical studies check a wall-time budget between completed cases. Check `planned`, `completed` and `complete` in their summaries. Repeat the same command with unchanged source/configuration to finish remaining cases. A successful pass can still contain budget-limited incomplete sets. Report individual available grid cases explicitly until the whole grid set is complete.

To summarize an existing run without modifying its numerical inputs:

```powershell
python -B additional_analysis/summarize_saved_analyses.py `
  --ablation-results Data/additional_analysis_results/ablation `
  --extended-results Data/additional_analysis_results/extended `
  --output Data/additional_analysis_summaries
```

`SOURCE_IDENTITIES.json` records input and public source hashes for the numerical components. Publication edits change package/default output names and administrative text, while preserving numerical algorithms and configuration values. Small artificial fixtures check the RF picker, support transforms, normalization, interpolation, physical refinement engines and protocol/resume behavior. Full scientific numerical outputs are produced by running the supplied commands with the retained input files.
