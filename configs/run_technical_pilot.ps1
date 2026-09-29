# TECHNICAL PILOT: 24 synthetic samples, 96x96 grid, one training epoch per stage.
# No pilot checkpoint is used by the full run. This tests execution, not image quality.
param(
  [string]$OutputRoot = "outputs/pilot",
  [string]$InputDataRoot = "data/ali_et_al",
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function OK([string]$Name) {
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
  Write-Host "[OK] $Name" -ForegroundColor Green
}
function Need([string]$PathName) {
  if (-not (Test-Path -LiteralPath $PathName -PathType Leaf)) { throw "Missing required file: $PathName" }
}

# Run from the repository root.
$ROOT = $OutputRoot
$TRAIN = "$ROOT/training"
$RUN = "$ROOT/measured_case"
$DEVICE = $Device

$DATA = "$TRAIN/ultrasound_data_24_pairs.pt"
$SPLITS = "$TRAIN/ultrasound_data_24_pairs_splits.pt"
$SOS_BANK = "$TRAIN/ultrasound_data_24_pairs_sos_bank.pt"
$BASE = "$TRAIN/reconstruction_model_supervised_roi_tof_best.pth"
$AE = "$TRAIN/latent_sos_autoencoder_v1.pth"
$OP = "$TRAIN/matched_tof_operator_v1.pth"

$STAGE4 = "$TRAIN/adjoint_latent_background_clean.pt"
$STAGE7 = "$TRAIN/spatial_physics_reconstructor_clean.pt"

$MAT_RF = "$InputDataRoot/Malignancy.mat"
$MAT_GT = "$InputDataRoot/Malignancy_WaveformInversionResults.mat"
$env:UST_GEOMETRY_MAT = $MAT_RF
$RAW = "$RUN/experimental_raw.pt"
$ALIGNED = "$RUN/experimental_aligned.pt"
$WITH_GT = "$RUN/experimental_aligned_with_gt.pt"
$EXP_SETUP = "$RUN/experimental_setup.pt"
$RAY_SETUP = "$RUN/physical_setup.pt"
$RECON = "$RUN/physical_reconstruction.pt"
$STAGE8_REPORT = "$TRAIN/synthetic_evaluation/synthetic_evaluation_report.npz"
$PHYSICAL_REPORT = "$RUN/physical_report.npz"

$RUN_STAGE_0 = $true
$RUN_STAGE_1 = $true
$RUN_STAGE_2 = $true
$RUN_STAGE_3 = $true
$RUN_STAGE_4 = $true
$RUN_STAGE_7 = $true
$RUN_STAGE_8 = $true
$RUN_STAGE_9 = $true
$RUN_STAGE_10 = $true

@(
  $MAT_RF,$MAT_GT,
  "src/train_initial_reconstruction.py","src/eikonal_forward_model.py","src/train_latent_sos_autoencoder.py","src/train_matched_tof_operator.py",
  "src/train_background_refiner.py","src/train_spatial_physics.py","src/evaluate_synthetic_reconstruction.py",
  "src/plot_synthetic_evaluation.py","src/extract_measured_tof.py",
  "src/attach_evaluation_reference.py","src/build_experimental_setup.py","src/build_physical_ray_setup.py",
  "src/reconstruct_physical_ray.py","src/plot_physical_ray.py","src/self_test.py","src/measured_geometry.py","src/align_experimental.py","src/logger.py","src/settings.py"
) | ForEach-Object { Need $_ }
if (-not $RUN_STAGE_0) { Need $DATA; Need $SPLITS }
if (-not $RUN_STAGE_1) { Need $BASE }
if (-not $RUN_STAGE_2) { Need $AE }
if (-not $RUN_STAGE_3) { Need $OP }
if (-not $RUN_STAGE_4) { Need $STAGE4 }
if (-not $RUN_STAGE_7) { Need $STAGE7 }

if (-not (Select-String -LiteralPath "src/train_background_refiner.py" -Pattern "background-refiner-v1" -Quiet)) {
  throw "The clean Stage 4 source was not installed in repository src."
}
if (-not (Select-String -LiteralPath "src/train_spatial_physics.py" -Pattern "spatial-physics-reconstructor-v1" -Quiet)) {
  throw "The clean Stage 7 source was not installed in repository src."
}
if (-not (Select-String -LiteralPath "src/build_physical_ray_setup.py" -Pattern "measured-coordinate-single-calibration-ray-setup-v1" -Quiet)) {
  throw "Ray-setup source signature mismatch."
}
if (-not (Select-String -LiteralPath "src/align_experimental.py" -Pattern "measured-geometry-single-gated-additive-calibration-v1" -Quiet)) {
  throw "Channel-alignment source signature mismatch."
}
if (-not (Select-String -LiteralPath "src/reconstruct_physical_ray.py" -Pattern "measured-coordinate-heldout-gated-physical-ray-v1" -Quiet)) {
  throw "Physical-reconstruction source signature mismatch."
}
Write-Host "[PREFLIGHT] Full measured-geometry pipeline; stages 0,1,2,3,4,7,8,9,10 enabled." -ForegroundColor Cyan
python src/self_test.py --geometry_stamp "$TRAIN/measured_geometry_sha256.txt"
OK "Geometry and calibration self-test"

New-Item -ItemType Directory -Force -Path $RUN,$TRAIN | Out-Null

if ($RUN_STAGE_0) {
  python src/train_initial_reconstruction.py --pairs_chunk_size 24 --pairs_workers 2 --mode train --data_path "$DATA" --sos_bank_path "$SOS_BANK" --splits_path "$SPLITS" --model_path "$TRAIN/_cache_build_dummy_model.pth" --output_dir "$TRAIN/synthetic_cache/" --device "$DEVICE" --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --radius 0.1091 --n_emitters 512 --n_receivers 512 --num_samples 24 --data_seed 52 --make_sos_bank --make_splits --make_pairs_cache --store_collocation_T --n_collocation 32 --colloc_emitters_k 4 --colloc_seed 52 --tof_norm zscore --exclude_frac 0.25 --synthetic_roi_tof_enable --synthetic_roi_radius_min 0.055 --synthetic_roi_radius_max 0.070 --synthetic_roi_outer_c_min 1460 --synthetic_roi_outer_c_max 1500 --synthetic_roi_reference_c 1500 --synthetic_roi_synthetic_c 1500 --synthetic_roi_min_chord_frac 0.05 --synthetic_roi_residual_clip_us 8.0 --synthetic_roi_nonintersect_mode reference --synthetic_roi_seed 52 --use_random_support_envelope --use_circular_mask --mask_radius 0.1091 --shape_mode mixed --min_shapes 8 --max_shapes 24 --p_ellipse 0.25 --p_triangle 0.25 --p_polygon 0.25 --p_rod 0.25 --shape_size_min_px 4 --shape_size_max_px 16 --polygon_vertices_min 4 --polygon_vertices_max 8
  OK "Stage 0 synthetic cache"
}

if ($RUN_STAGE_1) {
  python src/train_initial_reconstruction.py --mode train --data_path "$DATA" --sos_bank_path "$SOS_BANK" --splits_path "$SPLITS" --model_path "$BASE" --output_dir "$TRAIN/initial_reconstruction/" --device "$DEVICE" --resume --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --radius 0.1091 --n_emitters 512 --n_receivers 512 --num_samples 24 --data_seed 52 --model_type deep --latent_res 16 --model_fc_dropout 0.08 --model_decoder_dropout 0.04 --epochs 1 --batch_size 2 --lr 0.00025 --weight_decay 0.0001 --lambda_tof 0 --lambda_grad 0.08 --lambda_detail 0.35 --lambda_laplacian 0.02 --detail_edge_gain 4 --use_sos_weighted_loss --sos_weight_water 1 --sos_weight_contrast_gain 8 --sos_weight_power 1 --use_circular_mask --mask_radius 0.1091 --use_tof_mask_channel --tof_feature_mode residual_stack --tof_aug_enable --tof_aug_scale_min 0.92 --tof_aug_scale_max 1.08 --tof_aug_shift_std 0.10 --tof_aug_noise_std 0.015 --tof_feature_channel_dropout 0.08 --tof_emitter_dropout 0.02 --tof_receiver_dropout 0.02 --use_ema --ema_decay 0.995 --preview_every 0 --early_stop_patience 24 --early_stop_min_delta 0.00002 --early_stop_metric val_sos_detail
  OK "Stage 1 initial reconstruction"
}

if ($RUN_STAGE_2) {
  python src/train_latent_sos_autoencoder.py --data_path "$DATA" --splits_path "$SPLITS" --ae_path "$AE" --output_dir "$TRAIN/autoencoder/" --device "$DEVICE" --require_cuda --resume --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --epochs 1 --batch_size 2 --lr 0.0002 --weight_decay 0.0001 --latent_ch 64 --latent_grid 16 --channels 64 --dropout 0.02 --lambda_l1 1 --lambda_mse 0.7 --lambda_grad 0.5 --lambda_tv 0.002 --early_stop_patience 8 --early_stop_min_delta 0.00001 --preview_every 0 --batch_log_every 50 --use_circular_mask --mask_radius 0.1091
  OK "Stage 2 latent SoS autoencoder"
}

if ($RUN_STAGE_3) {
  python src/train_matched_tof_operator.py --data_path "$DATA" --splits_path "$SPLITS" --operator_path "$OP" --output_dir "$TRAIN/tof_operator/" --device "$DEVICE" --require_cuda --resume --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --n_emitters 512 --n_receivers 512 --epochs 1 --batch_size 2 --lr 0.0002 --weight_decay 0.0001 --operator_channels 24 --operator_latent_dim 96 --dropout 0.04 --lambda_l1 1 --lambda_mse 0.4 --early_stop_patience 14 --early_stop_min_delta 0.00001 --preview_every 0 --batch_log_every 50 --residual_plot_clip_us 8 --use_circular_mask --mask_radius 0.1091
  OK "Stage 3 matched ToF operator"
}

if ($RUN_STAGE_4) {
  python src/train_background_refiner.py --data_path "$DATA" --splits_path "$SPLITS" --base_model_path "$BASE" --ae_path "$AE" --operator_path "$OP" --reconstructor_path "$STAGE4" --output_dir "$TRAIN/background_refiner/" --device "$DEVICE" --require_cuda --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --radius 0.1091 --n_emitters 512 --n_receivers 512 --model_type deep --latent_res 16 --use_tof_mask_channel --tof_feature_mode residual_stack --epochs 1 --batch_size 2 --gradient_accumulation 3 --lr 0.00009 --weight_decay 0.0001 --channels 32 --dropout 0.05 --delta_limit 0.65 --adjoint_ray_samples 32 --adjoint_ray_chunk 2048 --adjoint_blur_kernel 9 --lowpass_kernel 21 --detail_kernel 7 --lambda_l1 1 --lambda_mse 0.65 --lambda_grad 0.55 --lambda_detail 0.80 --lambda_base_low_anchor 0.015 --lambda_base_full_anchor 0.005 --lambda_tof_l1 0.10 --lambda_tof_mse 0.03 --lambda_tv 0.001 --lambda_delta 0.002 --lambda_gate_sparsity 0.0005 --val_detail_weight 0.60 --val_no_improve_penalty 5 --max_train_samples 16 --max_val_samples 4 --augmentation_probability 0.75 --augmentation_global_us 0.35 --augmentation_emitter_us 0.30 --augmentation_receiver_us 0.30 --augmentation_noise_us 0.035 --augmentation_sector_drop_probability 0.20 --augmentation_channel_drop_probability 0.01 --augmentation_sectors 8 --augmentation_harmonics 3 --lambda_corrupt_reconstruction 0.45 --lambda_clean_corrupt_consistency 0.20 --early_stop_patience 3 --early_stop_min_delta 0.00002 --stop_if_not_better_than_base_patience 3 --preview_every 0 --batch_log_every 50 --residual_plot_clip_us 8 --tof_plot_fill finite_median --use_circular_mask --mask_radius 0.1091
  OK "Stage 4 acquisition-robust background fine-tuning"
}

if ($RUN_STAGE_7) {
  Need $STAGE4
  python src/train_spatial_physics.py --data_path "$DATA" --splits_path "$SPLITS" --base_model_path "$BASE" --ae_path "$AE" --operator_path "$OP" --background_reconstructor_path "$STAGE4" --model_path "$STAGE7" --output_dir "$TRAIN/spatial_physics/" --device "$DEVICE" --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --radius 0.1091 --n_emitters 512 --n_receivers 512 --model_type deep --latent_res 16 --use_tof_mask_channel --tof_feature_mode residual_stack --use_circular_mask --mask_radius 0.1091 --epochs 1 --batch_size 2 --lr 0.00012 --weight_decay 0.00008 --channels 32 --dropout 0.04 --delta_limit 0.55 --physics_residual_norm_scale 0.35 --background_delta_limit 0.65 --adjoint_ray_samples 32 --num_emitter_sectors 8 --num_receiver_sectors 8 --sector_highpass --lambda_res_l1 1.4 --lambda_res_detail 1.2 --lambda_final_mse 0.75 --lambda_final_l1 0.35 --lambda_tof 0.25 --lambda_tv 0.00035 --lambda_view_consistency 0.20 --lambda_stripe_sensitivity 0 --nuisance_global_delay_us 0.30 --nuisance_emitter_delay_us 0.25 --nuisance_receiver_delay_us 0.25 --nuisance_noise_us 0.030 --nuisance_sector_drop_prob 0.20 --nuisance_channel_drop_prob 0.01 --nuisance_harmonics 3 --directional_artifact_probability 0 --directional_artifact_amplitude 0 --max_train_samples 16 --max_val_samples 4 --early_stop_patience 3 --early_stop_min_delta 0.00002 --preview_count 0 --batch_log_every 20 --checkpoint_every_batches 200
  OK "Stage 7 short adaptation to corrected Stage 4"
}

if ($RUN_STAGE_8) {
  Need $STAGE7
  python src/evaluate_synthetic_reconstruction.py --data_path "$DATA" --splits_path "$SPLITS" --base_model_path "$BASE" --ae_path "$AE" --operator_path "$OP" --background_reconstructor_path "$STAGE4" --model_path "$STAGE7" --output_dir "$TRAIN/synthetic_evaluation/" --report_npz "$STAGE8_REPORT" --device "$DEVICE" --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --radius 0.1091 --n_emitters 512 --n_receivers 512 --model_type deep --latent_res 16 --use_tof_mask_channel --tof_feature_mode residual_stack --use_circular_mask --mask_radius 0.1091 --batch_size 4 --preview_count 0 --use_val_split --max_test_samples 4
  OK "Stage 8 numerical synthetic validation"
  $env:MPLBACKEND = "Agg"
  python src/plot_synthetic_evaluation.py --report_npz "$STAGE8_REPORT" --output_dir "$TRAIN/synthetic_evaluation/report/"
  OK "Stage 8 report"
}

if ($RUN_STAGE_9) {
  python src/extract_measured_tof.py --data_path "$DATA" --mat_path "$MAT_RF" --output_dir "$RUN/waveform_log/" --out_pt "$RAW" --no_subtract_min_per_emitter --ring_order_mode measured_geometry --t_skip 2e-5 --env_smooth 9 --thresh_rel 0.15 --exclude_frac 0.25 --gaussian_scale 0.5 --nan_fill row_median --tof_variant_to_save geom_window_matched_template --peak_window_pre_frac 0.05 --peak_window_post_frac 0.15 --matched_template_count 256 --matched_template_pre_samples 48 --matched_template_post_samples 96 --matched_template_min_correlation 0.25 --matched_template_max_lag_samples 12
  OK "Stage 9a experimental RF ToF extraction"
  python src/align_experimental.py --raw_pt "$RAW" --synthetic_data_path "$DATA" --out_pt "$ALIGNED" --report_json "$RUN/alignment.json" --output_dir "$RUN/alignment_log/" --synthetic_samples 12 --outer_quantile 0.20 --component_limit_us 0.75 --calibration_min_gain_us 0.02 --sos_min 1400 --sos_max 1650
  OK "Stage 9b channel alignment and electronics calibration"
  python src/attach_evaluation_reference.py --mat_path "$MAT_GT" --tof_pt "$ALIGNED" --out_pt "$WITH_GT" --output_dir "$RUN/gt_log/" --gt_iter 30 --skip_tof_diagnostic --skip_orientation_diagnostic
  OK "Stage 9c attach diagnostic GT without changing its orientation"
  python src/build_experimental_setup.py --data_path "$WITH_GT" --base_model_path "$BASE" --ae_path "$AE" --operator_path "$OP" --background_reconstructor_path "$STAGE4" --model_path "$STAGE7" --output_dir "$RUN/experimental_prior_log/" --out_pt "$EXP_SETUP" --device "$DEVICE" --angular_sectors 8 --support_taper_pixels 7 --support_radius_quantile 0.70 --model_type deep --latent_res 16 --use_tof_mask_channel --tof_feature_mode residual_stack --use_circular_mask --mask_radius 0.1091 --nx 96 --ny 96 --phys_x 0.24 --phys_y 0.24 --radius 0.1091 --n_emitters 512 --n_receivers 512
  OK "Stage 9d experimental prior"
  python src/build_physical_ray_setup.py --aligned_gt_pt "$WITH_GT" --support_setup_pt "$EXP_SETUP" --out_pt "$RAY_SETUP" --output_dir "$RUN/ray_setup_log/" --grid_size 64 --ray_samples 48 --emitter_stride 8 --receiver_stride 8 --ring_radius_m 0.1091 --phys_x 0.24 --phys_y 0.24 --water_sos 1500 --outer_margin_m 0.003
  OK "Stage 9e measured-coordinate ray cache (no second calibration)"
}

if ($RUN_STAGE_10) {
  Need $RAY_SETUP
  python src/reconstruct_physical_ray.py --setup_pt "$RAY_SETUP" --out_pt "$RECON" --report_npz "$PHYSICAL_REPORT" --output_dir "$RUN/reconstruction_log/" --device "$DEVICE" --holdout_sectors "0,2,4,6" --angular_sectors 8 --steps 4 --lr 0.035 --max_correction_mps 120 --huber_beta_us 0.20 --prior_weight 0.035 --tv_weight 0.020 --curvature_weight 0.008 --warmup 1 --patience 4 --ray_chunk 2048 --min_holdout_improvement_us 0.01 --min_accepted_folds 2
  OK "Stage 10a physical-ray reconstruction"
  $env:MPLBACKEND = "Agg"
  python src/plot_physical_ray.py --report_npz "$PHYSICAL_REPORT" --output_dir "$RUN/report/"
  OK "Stage 10b final report"
}

Write-Host "Pipeline completed successfully under $RUN" -ForegroundColor Green







