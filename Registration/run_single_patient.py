"""
Run the full non-linear pipeline for a SINGLE patient and save
individual atlas registration images so you can visually see which
BSpline registrations worked and which fell back to affine.

Usage:
    python run_single_patient.py              # defaults to patient 96
    python run_single_patient.py 117          # run patient 117
    python run_single_patient.py 96 10        # run patient 96, show slice 10
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re
import sys
import time
import math
import numpy as np

from prostateLoader import ProstateLoader
from utils import final_metric_from_elastix_log

import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration — change these as needed
# ============================================================
TARGET_PATIENT = int(sys.argv[1]) if len(sys.argv) > 1 else 96
VIS_SLICE = int(sys.argv[2]) if len(sys.argv) > 2 else 15

ATLAS_SIZE = 64
PRESELECTION_SIZE = 5        # top 5 for BSpline (matches top5_fast)
FUSION_SIZE = 5              # majority vote over 5
BSPLINE_BLACKLIST = {34}

SAVE_DIR = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results")

AFFINE_PARAM = str(SAVE_DIR / "affine.txt")
BSPLINE_PARAM = str(Path(r"C:\Users\30697\OneDrive - University of West Attica\Documents\GitHub\CSinMIA\ParameterFiles\BSpline\bspline.txt"))


# ============================================================
# Helper functions
# ============================================================
def safe_final_metric(max_retries=5, wait_sec=3):
    for attempt in range(max_retries):
        try:
            return final_metric_from_elastix_log()
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"    PermissionError, retry {attempt+1}/{max_retries}...")
                time.sleep(wait_sec)
            else:
                raise


def final_metric_from_log(log_path):
    pat = re.compile(r"Final metric value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    metric = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                metric = float(m.group(1))
    if metric is None:
        raise RuntimeError(f"No 'Final metric value' in {log_path}")
    return -metric


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
# Main
# ============================================================
print(f"Loading data...")
t0 = time.perf_counter()

loader = ProstateLoader()
images, segmen = loader.LoadData()

NUM_VOLUMES = len(images)
if TARGET_PATIENT >= NUM_VOLUMES:
    print(f"ERROR: Patient {TARGET_PATIENT} not found (max index: {NUM_VOLUMES-1})")
    sys.exit(1)

atlas_images = images[0:ATLAS_SIZE]
atlas_segmen = segmen[0:ATLAS_SIZE]

t_load = time.perf_counter()
print(f"Loaded in {t_load - t0:.1f}s. Running patient {TARGET_PATIENT}.")
print(f"Pipeline: {ATLAS_SIZE} affine -> top {PRESELECTION_SIZE} -> BSpline -> fusion")

pm_affine = sitk.ReadParameterFile(AFFINE_PARAM)

fixed_img = images[TARGET_PATIENT]
gt_mask = segmen[TARGET_PATIENT]

# ============================================================
# Stage 1: Affine registration (64 atlases)
# ============================================================
print(f"\n  Stage 1: Affine registering {ATLAS_SIZE} atlases...")
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
    if (idx + 1) % 16 == 0:
        print(f"    {idx+1}/{ATLAS_SIZE} done...")

t_aff_end = time.perf_counter()
print(f"  Affine done in {t_aff_end - t_aff_start:.1f}s")

# ============================================================
# Stage 2: Select top 5 by NCC
# ============================================================
ranked = sorted(range(ATLAS_SIZE), key=lambda k: metrics[k], reverse=True)
top_indices = ranked[:PRESELECTION_SIZE]
top_metrics = [metrics[i] for i in top_indices]
top_tmaps = [transforms[i] for i in top_indices]

print(f"\n  Top {PRESELECTION_SIZE} atlases: {top_indices}")
print(f"  NCC scores: {[f'{m:.4f}' for m in top_metrics]}")

# ============================================================
# Stage 3: BSpline refinement on top 5
# ============================================================
print(f"\n  Stage 3: BSpline refinement...")
t_bs_start = time.perf_counter()

bs_metrics = []
bs_tmaps = []
used_stage = []       # "BS" or "AFF" for each atlas
atlas_warped = []     # individual warped masks for visualization

with TemporaryDirectory(prefix="single_") as tmp:
    tmp_root = Path(tmp)

    for i, atlas_idx in enumerate(top_indices):
        work_dir = tmp_root / f"atlas_{atlas_idx:03d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write affine transform
        aff_file = work_dir / "affine_tp.txt"
        sitk.WriteParameterFile(last_parameter_map(top_tmaps[i]), str(aff_file))
        tp_text = aff_file.read_text(encoding="utf-8")
        tp_text = re.sub(
            r'\(InitialTransformParametersFileName\s+"[^"]*"\)',
            '(InitialTransformParametersFileName "NoInitialTransform")', tp_text)
        aff_file.write_text(tp_text, encoding="utf-8")

        if atlas_idx in BSPLINE_BLACKLIST:
            print(f"    Atlas {atlas_idx:03d}: BLACKLISTED -> AFF")
            bs_metrics.append(None)
            bs_tmaps.append(top_tmaps[i])
            used_stage.append("AFF")
            # Warp with affine
            warp_dir = tmp_root / f"warp_{atlas_idx:03d}"
            warped = warp_label(atlas_segmen[atlas_idx], top_tmaps[i], warp_dir)
            atlas_warped.append(warped)
            continue

        try:
            pm_bs = sitk.ReadParameterFile(BSPLINE_PARAM)
            pm_bs["InitialTransformParameterFileName"] = [str(aff_file).replace("\\", "/")]
            pm_bs["HowToCombineTransforms"] = ["Compose"]

            elx_bs = sitk.ElastixImageFilter()
            elx_bs.SetFixedImage(fixed_img)
            elx_bs.SetMovingImage(atlas_images[atlas_idx])
            elx_bs.SetParameterMap(pm_bs)
            elx_bs.SetOutputDirectory(str(work_dir))
            elx_bs.LogToConsoleOff()
            elx_bs.LogToFileOn()
            elx_bs.Execute()

            bs_metric = final_metric_from_log(work_dir / "elastix.log")

            if bs_metric > top_metrics[i]:
                print(f"    Atlas {atlas_idx:03d}: AFF={top_metrics[i]:.4f}  BS={bs_metric:.4f}  -> BS (improved)")
                bs_metrics.append(bs_metric)
                bs_tmaps.append(elx_bs.GetTransformParameterMap())
                used_stage.append("BS")
                # Warp with BSpline
                warp_dir = tmp_root / f"warp_{atlas_idx:03d}"
                warped = warp_label(atlas_segmen[atlas_idx], elx_bs.GetTransformParameterMap(), warp_dir)
                atlas_warped.append(warped)
            else:
                print(f"    Atlas {atlas_idx:03d}: AFF={top_metrics[i]:.4f}  BS={bs_metric:.4f}  -> AFF (no improvement)")
                bs_metrics.append(bs_metric)
                bs_tmaps.append(top_tmaps[i])
                used_stage.append("AFF")
                # Warp with affine
                warp_dir = tmp_root / f"warp_{atlas_idx:03d}"
                warped = warp_label(atlas_segmen[atlas_idx], top_tmaps[i], warp_dir)
                atlas_warped.append(warped)

        except Exception as e:
            print(f"    Atlas {atlas_idx:03d}: FAILED ({str(e)[:60]}) -> AFF")
            bs_metrics.append(None)
            bs_tmaps.append(top_tmaps[i])
            used_stage.append("AFF")
            warp_dir = tmp_root / f"warp_{atlas_idx:03d}"
            warped = warp_label(atlas_segmen[atlas_idx], top_tmaps[i], warp_dir)
            atlas_warped.append(warped)

    t_bs_end = time.perf_counter()
    print(f"  BSpline done in {t_bs_end - t_bs_start:.1f}s")

    # ============================================================
    # Stage 4: Fusion
    # ============================================================
    fused_mask = vote_fusion(atlas_warped)

# ============================================================
# Compute metrics
# ============================================================
dice = dice_score(fused_mask, gt_mask)
jacc = jaccard_score(fused_mask, gt_mask)
hd = hausdorff_distance_mm(fused_mask, gt_mask)
rvd = relative_volume_difference(fused_mask, gt_mask)
n_bs = sum(1 for s in used_stage if s == "BS")
t_total = time.perf_counter() - t0

print(f"\n{'='*60}")
print(f"  RESULTS — Patient {TARGET_PATIENT}")
print(f"{'='*60}")
print(f"  Dice:      {dice:.4f}")
print(f"  Jaccard:   {jacc:.4f}")
print(f"  Hausdorff: {hd:.2f} mm")
print(f"  RVD:       {rvd:.4f}")
print(f"  BSpline:   {n_bs}/{PRESELECTION_SIZE}")
print(f"  Time:      {t_total:.1f}s ({t_total/60:.1f}min)")

# ============================================================
# Save metrics JSON
# ============================================================
result = {
    "patient": TARGET_PATIENT,
    "dice": round(dice, 4),
    "jaccard": round(jacc, 4),
    "hausdorff": round(hd, 4),
    "rvd": round(rvd, 4),
    "time_s": round(t_total, 1),
    "bspline_used": n_bs,
    "atlases_used": top_indices,
    "stages": used_stage
}
json_path = SAVE_DIR / f"patient_{TARGET_PATIENT:03d}_single_run.json"
with open(json_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved metrics: {json_path}")

# ============================================================
# Visualization: individual atlas results + fused + GT
# ============================================================
fixed_np = sitk.GetArrayFromImage(fixed_img)
gt_np = sitk.GetArrayFromImage(binarize(gt_mask))
fused_np = sitk.GetArrayFromImage(binarize(fused_mask))

# Pick the best slice — use the one with most GT voxels if VIS_SLICE not good
gt_slices = gt_np.sum(axis=(1, 2))
best_slice = int(np.argmax(gt_slices))
sl = best_slice  # use the slice with most prostate area

print(f"  Using visualization slice: {sl} (most prostate area)")

# Figure: 5 individual atlases + fused + ground truth = 7 panels
n_panels = PRESELECTION_SIZE + 2  # 5 atlases + fused + GT
fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), dpi=120)

fig.suptitle(
    f"Patient {TARGET_PATIENT} — Non-Linear Pipeline ({n_bs}/{PRESELECTION_SIZE} BSpline)\n"
    f"Dice: {dice:.4f} | Jaccard: {jacc:.4f} | HD: {hd:.2f} mm | RVD: {rvd:.4f}",
    fontsize=14, y=1.02
)

# Individual atlas panels
for i in range(PRESELECTION_SIZE):
    ax = axes[i]
    atlas_idx = top_indices[i]
    stage = used_stage[i]
    warped_np = sitk.GetArrayFromImage(binarize(atlas_warped[i]))

    # Compute per-atlas dice
    p = warped_np.astype(np.uint8)
    g = gt_np.astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    atlas_dice = (2.0 * inter / den) if den > 0 else 0.0

    ax.imshow(fixed_np[sl], cmap="gray")
    mask_overlay = np.ma.masked_where(warped_np[sl] == 0, warped_np[sl])
    color = "Greens" if stage == "BS" else "Oranges"
    ax.imshow(mask_overlay, cmap=color, alpha=0.45)

    # Also overlay GT contour
    from matplotlib.colors import ListedColormap
    gt_contour = np.zeros_like(gt_np[sl], dtype=float)
    if gt_np[sl].sum() > 0:
        from scipy import ndimage
        eroded = ndimage.binary_erosion(gt_np[sl], iterations=1)
        gt_contour = (gt_np[sl].astype(float) - eroded.astype(float))
    contour_masked = np.ma.masked_where(gt_contour == 0, gt_contour)
    ax.imshow(contour_masked, cmap="Reds", alpha=0.8)

    label = f"Atlas {atlas_idx}\n{stage} | Dice: {atlas_dice:.3f}"
    ax.set_title(label, fontsize=11, pad=6,
                 color="green" if stage == "BS" else "darkorange",
                 fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

# Fused panel
ax = axes[PRESELECTION_SIZE]
ax.imshow(fixed_np[sl], cmap="gray")
fused_overlay = np.ma.masked_where(fused_np[sl] == 0, fused_np[sl])
ax.imshow(fused_overlay, cmap="Blues", alpha=0.45)
ax.imshow(contour_masked, cmap="Reds", alpha=0.8)
ax.set_title(f"Fused ({n_bs}/{PRESELECTION_SIZE} BS)\nDice: {dice:.4f}",
             fontsize=11, pad=6, color="blue", fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])

# Ground truth panel
ax = axes[PRESELECTION_SIZE + 1]
ax.imshow(fixed_np[sl], cmap="gray")
gt_overlay = np.ma.masked_where(gt_np[sl] == 0, gt_np[sl])
ax.imshow(gt_overlay, cmap="Reds", alpha=0.45)
ax.set_title("Ground Truth", fontsize=11, pad=6, color="red", fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.82, wspace=0.05)

fig_path = SAVE_DIR / f"patient_{TARGET_PATIENT:03d}_atlas_detail.png"
fig.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved figure: {fig_path}")

print(f"\nDone! Total time: {time.perf_counter() - t0:.1f}s")
