# Checkpoint release checklist

The publication source repository deliberately excludes large checkpoints and generated caches. Before public release, archive only the five final checkpoints and the split metadata required for measured inference:

- `reconstruction_model_supervised_roi_tof_best.pth`
- `latent_sos_autoencoder_v1.pth`
- `matched_tof_operator_v1.pth`
- `adjoint_latent_background_clean.pt`
- `spatial_physics_reconstructor_clean.pt`
- `ultrasound_data_20000_pairs_splits.pt`

Upload them to a durable research-data service and replace the placeholders below:

```text
Archive DOI: REPLACE_WITH_ARCHIVE_DOI
Archive URL: REPLACE_WITH_ARCHIVE_URL
```

Record a SHA-256 checksum for every released file. In PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 path/to/checkpoint
```

Do not publish superseded checkpoints, optimizer scratch files, cached debug tensors, or copies of the upstream Ali data.








