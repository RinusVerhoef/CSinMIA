"""
Batch pipeline: 64 affine-only -> top 5 -> fusion (NO BSpline)
Loops through all test patients (64-138).
Skips patients with existing output PNGs (resume capable).
Saves segmentation figures + metrics JSON to affine_64/
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re
import time
import math
import numpy as np

from prostateLoader import ProstateLoader
from utils import final_metric_from_elastix_log

import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_final_metric(max_retries=5, wait_sec=3):
    """Retry wrapper for final_metric_from_elastix_log — handles PermissionError."""
    for attempt in range(max_retries):
        try:
            return final_metric_from_elastix_log()
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"    PermissionError reading elastix.log, retry {attempt+1}/{max_retries}...")
                time.sleep(wait_sec)
            else:
                raise
    raise RuntimeError("Failed to read elastix.log after retries")


def last_parameter_map(tmap):
    try:
        return tmap[-1]
    except Exception:
        return tmap


def binarize(img):
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def dice_score(pred, gt):
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    return 1.0 if den == 0 else (2.0 * inter / den)


def jaccard_score(pred, gt):
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    union = int(((p | g) > 0).sum())
    return 1.0 if union == 0 else (inter / union)


def hausdorff_distance_mm(pred, gt):
    pred_b, gt_b = binarize(pred), binarize(gt)
    pred_sum = int(sitk.GetArrayFromImage(pred_b).sum())
    gt_sum = int(sitk.GetArrayFromImage(gt_b).sum())
    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return float('inf')
    hd = sitk.HausdorffDistanceImageFilter()
    hd.Execute(pred_b, gt_b)
    return float(hd.GetHausdorffDistance())


def relative_volume_difference(pred, gt):
    vp = float(sitk.GetArrayFromImage(binarize(pred)).sum())
    vg = float(sitk.GetArrayFromImage(binarize(gt)).sum())
    if vg == 0:
        return 0.0 if vp == 0 else float('inf')
    return (vp - vg) / vg


def vote_fusion(masks):
    arrs = [sitk.GetArrayFromImage(binarize(m)).astype(np.uint8) for m in masks]
    votes = np.stack(arrs, axis=0).sum(axis=0)
    threshold = math.ceil(len(masks) / 2)
    fused = (votes >= threshold).astype(np.uint8)
    fused_img = sitk.GetImageFromArray(fused)
    fused_img.CopyInformation(masks[0])
    return fused_img


def warp_label(label_img, tmap, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tp_file = out_dir / "tp.txt"
    sitk.WriteParameterFile(last_parameter_map(tmap), str(tp_file))
    tp_text = tp_file.read_text(encoding="utf-8")
    tp_text = re.sub(
        r'\(InitialTransformParametersFileName\s+"[^"]*"\)',
        '(InitialTransformParametersFileName "NoInitialTransform")', tp_text)
    tp_text = tp_text.replace(
        '(ResampleInterpolator "FinalBSplineInterpolator")',
        '(ResampleInterpolator "FinalNearestNeighborInterpolator")')
    tp_text = tp_text.replace(
        '(ResampleInterpolator "FinalLinearInterpolator")',
        '(ResampleInterpolator "FinalNearestNeighborInterpolator")')
    tp_file.write_text(tp_text, encoding="utf-8")
    tfx = sitk.TransformixImageFilter()
    tfx.SetMovingImage(label_img)
    tfx.SetTransformParameterMap(sitk.ReadParameterFile(str(tp_file)))
    tfx.SetOutputDirectory(str(out_dir))
    tfx.LogToConsoleOff()
    tfx.LogToFileOff()
    tfx.Execute()
    return sitk.Cast(tfx.GetResultImage() > 0.5, sitk.sitkUInt8)


# ============================================================
# Settings
# ============================================================
ATLAS_SIZE = 64
TEST_START = 64           # first test patient index
FUSION_SIZE = 5
VISUALISATION_SLICE = 15
SAVE_DIR = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results\affine_64")

# ============================================================
print("Loading data...")
t0_global = time.perf_counter()

loader = ProstateLoader()
images, segmen = loader.LoadData()

NUM_VOLUMES = len(images)
atlas_images = images[0:ATLAS_SIZE]
atlas_segmen = segmen[0:ATLAS_SIZE]

SAVE_DIR.mkdir(parents=True, exist_ok=True)

t_load = time.perf_counter()
print(f"Loaded in {t_load - t0_global:.1f}s.  Atlas: {ATLAS_SIZE}  Test patients: {TEST_START}-{NUM_VOLUMES-1}")
print("Mode: AFFINE ONLY (no BSpline)")

pm_affine = sitk.ReadParameterFile(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results\affine.txt")

all_results = []

for test_idx in range(TEST_START, NUM_VOLUMES):
    fig_path = SAVE_DIR / f"patient_{test_idx:03d}_segmentation.png"
    if fig_path.exists():
        print(f"\n--- Patient {test_idx}: SKIPPED (already exists) ---")
        continue

    print(f"\n{'='*60}")
    print(f"  PATIENT {test_idx}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    fixed_img = images[test_idx]
    gt_mask = segmen[test_idx]

    # --- Affine registration ---
    print(f"  Affine registering {ATLAS_SIZE} atlases...")
    t_aff_start = time.perf_counter()

    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed_img)
    elx.SetParameterMap(pm_affine)
    elx.LogToConsoleOff()
    elx.LogToFileOn()

    metrics = []
    transforms = []

    for idx, moving_img in enumerate(atlas_images):
        elx.SetMovingImage(moving_img)
        elx.Execute()
        tmap = elx.GetTransformParameterMap()
        metric = safe_final_metric()
        metrics.append(metric)
        transforms.append(tmap)

    t_aff_end = time.perf_counter()
    print(f"  Affine done in {t_aff_end - t_aff_start:.1f}s  ({(t_aff_end - t_aff_start)/ATLAS_SIZE:.1f}s per atlas)")

    # --- Select top 5 by affine metric ---
    ranked = sorted(range(ATLAS_SIZE), key=lambda k: metrics[k], reverse=True)
    top_indices = ranked[:FUSION_SIZE]
    top_metrics = [metrics[i] for i in top_indices]
    top_tmaps = [transforms[i] for i in top_indices]

    print(f"  Top {FUSION_SIZE} atlases: {top_indices}")
    print(f"  Top metrics: {[f'{m:.4f}' for m in top_metrics]}")

    # --- Warp masks and fuse (affine transforms only) ---
    print(f"  Fusion (top {FUSION_SIZE})...")
    t_fuse_start = time.perf_counter()

    with TemporaryDirectory(prefix="affine_") as tmp:
        tmp_root = Path(tmp)
        warped_masks = []
        for i, atlas_idx in enumerate(top_indices):
            warp_dir = tmp_root / f"warp_{atlas_idx:03d}"
            warped = warp_label(atlas_segmen[atlas_idx], top_tmaps[i], warp_dir)
            warped_masks.append(warped)

        fused_mask = vote_fusion(warped_masks)

    t_fuse_end = time.perf_counter()
    print(f"  Fusion done in {t_fuse_end - t_fuse_start:.1f}s")

    # --- Metrics ---
    dice = dice_score(fused_mask, gt_mask)
    jacc = jaccard_score(fused_mask, gt_mask)
    hd = hausdorff_distance_mm(fused_mask, gt_mask)
    rvd = relative_volume_difference(fused_mask, gt_mask)

    t_total = time.perf_counter() - t0

    result = {
        "patient": test_idx,
        "dice": round(dice, 4),
        "jaccard": round(jacc, 4),
        "hausdorff": round(hd, 4),
        "rvd": round(rvd, 4),
        "time_s": round(t_total, 1),
        "method": "affine_only",
        "atlases_used": top_indices
    }
    all_results.append(result)

    print(f"\n  Dice: {dice:.4f}  Jaccard: {jacc:.4f}  HD: {hd:.2f}mm  RVD: {rvd:.4f}")
    print(f"  Time: {t_total:.1f}s ({t_total/60:.1f}min)")

    # --- Save metrics JSON ---
    json_path = SAVE_DIR / f"patient_{test_idx:03d}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # --- Save segmentation figure ---
    fixed_np = sitk.GetArrayFromImage(fixed_img)
    fused_np = sitk.GetArrayFromImage(binarize(fused_mask))
    gt_np = sitk.GetArrayFromImage(binarize(gt_mask))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=120)
    fig.suptitle(
        f"Patient {test_idx} — {ATLAS_SIZE} atlases, affine only, top {FUSION_SIZE} fusion\n"
        f"Dice: {dice:.3f} | Jaccard: {jacc:.3f} | "
        f"Hausdorff: {hd:.3f} mm | RVD: {rvd:.3f}", fontsize=14, y=0.97)
    axes[0].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
    axes[0].set_title("Test image", fontsize=13, pad=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
    axes[1].imshow(np.ma.masked_where(fused_np[VISUALISATION_SLICE]==0,
                    fused_np[VISUALISATION_SLICE]), cmap="Reds", alpha=0.4)
    axes[1].set_title("Fused prediction", fontsize=13, pad=8)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
    axes[2].imshow(np.ma.masked_where(gt_np[VISUALISATION_SLICE]==0,
                    gt_np[VISUALISATION_SLICE]), cmap="Reds", alpha=0.4)
    axes[2].set_title("Ground truth mask", fontsize=13, pad=8)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.06, top=0.80, wspace=0.03)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

# ============================================================
# Summary
# ============================================================
if all_results:
    dices = [r["dice"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {len(all_results)} patients processed")
    print(f"{'='*60}")
    print(f"  Mean Dice:   {np.mean(dices):.4f}")
    print(f"  Median Dice: {np.median(dices):.4f}")
    print(f"  Min Dice:    {min(dices):.4f}")
    print(f"  Max Dice:    {max(dices):.4f}")
    print(f"  Total time:  {time.perf_counter() - t0_global:.0f}s ({(time.perf_counter() - t0_global)/60:.1f}min)")
else:
    print("\nAll patients already processed — nothing to do.")
