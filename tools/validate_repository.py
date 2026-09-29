#!/usr/bin/env python3
"""Static completeness and syntax check for the publication package."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "LICENSE", "CITATION.cff", "environment.yml", "requirements.txt",
    "configs/run_full.ps1", "configs/run_technical_pilot.ps1",
    "configs/run_measured_inference_from_checkpoints.ps1",
    "configs/reported_configuration.yaml",
    "src/train_initial_reconstruction.py", "src/train_background_refiner.py", "src/train_spatial_physics.py",
    "src/evaluate_synthetic_reconstruction.py", "src/extract_measured_tof.py", "src/align_experimental.py",
    "src/measured_geometry.py", "src/self_test.py", "src/reporting/__init__.py",
    "src/build_experimental_setup.py", "src/build_physical_ray_setup.py",
    "src/reconstruct_physical_ray.py",
]


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing publication files:\n  " + "\n  ".join(missing))
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    print(f"[OK] Publication package complete; Python syntax checked under {ROOT}")


if __name__ == "__main__":
    main()








