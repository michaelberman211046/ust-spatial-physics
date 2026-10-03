# Spatial Physics-Informed Neural Reconstruction for Ultrasound Travel-Time Tomography

This repository is the software package accompanying the manuscript. It contains the source code and recorded experimental configuration used for synthetic-only training, measured radio-frequency (RF) preprocessing, travel-time extraction, learned reconstruction, held-out-channel physical refinement, and numerical validation.

## Scientific scope

All trainable neural-network parameters are learned from synthetic examples. The measured case is not used to train or fine-tune those networks. The measured RF data are converted to time of flight (ToF), aligned to the acquisition geometry, passed through the frozen learned stack, and then processed by bounded straight-ray and Eikonal refinements.

The waveform-inversion reconstruction distributed by Ali et al. is attached only for retrospective evaluation and display. It is not supplied to the learned model, timing calibration objective, support estimation, or physical-refinement objective.

For both synthetic and measured reconstructions, the post-inference Eikonal calculation uses a coarse correction followed by an edge-aware fine correction. Angularly withheld transmitter sectors determine whether a proposed correction is retained. The measured-data parameterization also removes the spatially uniform correction mode and represents a bounded common timing residual separately from the image update.

## Repository layout

- `src/`: only the active publication pipeline modules and their direct local dependencies.
- `configs/run_full.ps1`: exact end-to-end command used for the reported run.
- `configs/run_technical_pilot.ps1`: reduced technical smoke test; not a scientific reproduction.
- `configs/run_measured_inference_from_checkpoints.ps1`: measured RF-to-ToF processing, frozen-model inference, and physical refinement using trained checkpoints.
- `configs/reported_configuration.yaml`: readable summary of the reported configuration.
- `tools/validate_repository.py`: verifies package completeness and Python syntax.
- `docs/`: data-acquisition, checkpoint-release, and provenance documentation.

No checkpoints, generated datasets, patient-derived data, temporary output, debug scripts, or obsolete development versions are included.

## Software environment

Python 3.10 or 3.11 and an NVIDIA CUDA-capable PyTorch installation are recommended for training. Create the portable environment with:

```powershell
conda env create -f environment.yml
conda activate ust-spatial-physics
```

The `environment.yml` deliberately installs PyTorch from the official PyTorch channel without hard-coding a CUDA build. Select the PyTorch/CUDA combination supported by the target GPU and driver. `requirements.txt` lists the direct Python dependencies for non-Conda environments.

## Obtain the public Ali et al. data

The measured RF data and the waveform-inversion comparison are provided by the upstream `WaveformInversionUST` project:

```powershell
git clone https://github.com/rehmanali1994/WaveformInversionUST external/WaveformInversionUST
```

The upstream project and data are described in:

> R. Ali et al., “2-D Slicewise Waveform Inversion of Sound Speed and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU Solver,” IEEE Transactions on Medical Imaging, 43(8), 2988--3000, 2024. DOI: 10.1109/TMI.2024.3383816.

From the upstream material, place the required files at:

```text
data/ali_et_al/Malignancy.mat
data/ali_et_al/Malignancy_WaveformInversionResults.mat
```

If the second file has a different upstream name, copy or rename the corresponding malignancy waveform-inversion result to the path above; do not transform or reorient it. The upstream data and code remain governed by their own license and citation requirements and are not redistributed here. See [docs/DATA.md](docs/DATA.md).

## Reproduce the complete reported pipeline

Open PowerShell in the repository root. Confirm that the two Ali data files exist under `data/ali_et_al`, then run:

```powershell
powershell -ExecutionPolicy Bypass -File configs/run_full.ps1
```

The defaults write generated artifacts to `outputs/`, read the Ali et al. files
from `data/ali_et_al/`, and use CUDA. These locations are configurable:

```powershell
powershell -ExecutionPolicy Bypass -File configs/run_full.ps1 `
  -OutputRoot "path/to/results" `
  -InputDataRoot "path/to/ali_data" `
  -Device "cuda"
```

The full run creates 20,000 synthetic examples, trains all learned components, evaluates the synthetic validation subset, applies coarse and fine synthetic Eikonal refinement, extracts measured ToFs, performs measured-data inference, and applies the straight-ray and two-resolution Eikonal refinements. It is computationally expensive. The exact command file is the authoritative hyperparameter record.

To check installation and data flow on a small workload first:

```powershell
powershell -ExecutionPolicy Bypass -File configs/run_technical_pilot.ps1
```

The pilot confirms technical execution only and must not be used for manuscript results.

## Reproduce measured-data inference from published checkpoints

Place the following trained checkpoints under `outputs/training/`:

```text
reconstruction_model_supervised_roi_tof_best.pth
latent_sos_autoencoder_v1.pth
matched_tof_operator_v1.pth
adjoint_latent_background_clean.pt
spatial_physics_reconstructor_clean.pt
ultrasound_data_20000_pairs.pt
ultrasound_data_20000_pairs_splits.pt
```

Then run:

```powershell
powershell -ExecutionPolicy Bypass -File configs/run_measured_inference_from_checkpoints.ps1
```

The large checkpoints and synthetic cache are intentionally excluded from Git. A public archival release should attach them through Zenodo or an equivalent research-data repository and record checksums in `docs/CHECKPOINTS.md`.

## Verification

```powershell
python tools/validate_repository.py
```

This checks the expected publication files and compiles every Python source file without executing the training pipeline.

## License and citation

Original code in this repository is released under the MIT License; see `LICENSE`. Third-party data and software are excluded from that grant. Citation metadata are provided in `CITATION.cff`.








