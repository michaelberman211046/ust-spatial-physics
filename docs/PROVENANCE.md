# Source and result provenance

- Computational pipeline source: `src/`.
- Authoritative experimental configuration: `configs/run_full.ps1`.

`evaluate_synthetic_reconstruction.py` exports multiple synthetic test reconstructions using the frozen Stage-7 checkpoint and evaluation path. Report generation does not train or modify model weights.

The package intentionally omits:

- superseded experiments and inactive pipeline stages;
- debugging utilities;
- intermediate checkpoints and optimizer scratch files;
- generated tensors, figures, HTML logs, and temporary outputs;
- the Ali et al. upstream data;
- unrelated presentation and manuscript-development files.

`SHA256SUMS.txt` records the contents of this assembled publication snapshot.








