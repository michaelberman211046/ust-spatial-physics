# Data acquisition and placement

## Ali et al. measured data

The measured RF data and FWI comparison originate from the public repository:

https://github.com/rehmanali1994/WaveformInversionUST

The corresponding paper is available at DOI `10.1109/TMI.2024.3383816`; the repository README also links a versioned Zenodo archive.

Clone the upstream repository outside this repository's tracked files:

```powershell
git clone https://github.com/rehmanali1994/WaveformInversionUST external/WaveformInversionUST
```

Copy the malignancy RF acquisition and its waveform-inversion result into:

```text
data/ali_et_al/Malignancy.mat
data/ali_et_al/Malignancy_WaveformInversionResults.mat
```

The first file is processed to obtain ToF. The second is used only as an evaluation reference and is attached after the measured ToF has been prepared. It must not be used to choose timing calibration, anatomical support, model weights, acceptance thresholds, or physical-refinement parameters.

Upstream files are not redistributed here. Preserve their license and cite Ali et al.

## Synthetic data

Synthetic examples are generated locally by Stage 0. They are not downloaded and are excluded from Git because the cache is large. The exact generator arguments are in `configs/run_full.ps1`.

## Expected result tree

```text
outputs/
  training/
    ultrasound_data_20000_pairs.pt
    ultrasound_data_20000_pairs_splits.pt
    ultrasound_data_20000_pairs_sos_bank.pt
    reconstruction_model_supervised_roi_tof_best.pth
    latent_sos_autoencoder_v1.pth
    matched_tof_operator_v1.pth
    adjoint_latent_background_clean.pt
    spatial_physics_reconstructor_clean.pt
    synthetic_evaluation/synthetic_evaluation_report.npz
  measured_case/
    experimental_aligned.pt
    physical_report.npz
```








