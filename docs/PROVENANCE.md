# Source and result provenance

- Computational pipeline source: `src/`.
- Authoritative experimental configuration: `configs/run_full.ps1`.

`evaluate_synthetic_reconstruction.py` exports multiple synthetic validation reconstructions using the frozen learned stack. `refine_synthetic_eikonal_coarse.py` and `refine_synthetic_eikonal_fine.py` perform the subsequent held-out-sector Eikonal corrections. The measured reconstruction follows the straight-ray result with `refine_measured_eikonal_coarse.py` and `refine_measured_eikonal_fine.py`. None of these post-inference operations trains or modifies model weights.

The package intentionally omits:

- superseded experiments and inactive pipeline stages;
- debugging utilities;
- intermediate checkpoints and optimizer scratch files;
- generated tensors, figures, HTML logs, and temporary outputs;
- the Ali et al. upstream data;
- unrelated presentation and manuscript-development files.

`SHA256SUMS.txt` records the contents of this assembled publication snapshot.








