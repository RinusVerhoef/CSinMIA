"""
Single patient: 64 affine -> top 5 -> BSpline (original) -> best 5 -> fusion
Change TEST_PATIENT to run a different patient.
Shows all 5 atlas registered images with affine/BSpline metrics.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
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


def latest_tp(out_dir):
    tps = sorted(Path(out_dir).glob("TransformParameters.*.txt"))
    if not tps:
        raise FileNotFoundError(f"No TransformParameters.*.txt in {out_dir}")
    return tps[-1]


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


def warp_image(moving_img, tmap, out_dir):
    """Warp an IMAGE (not label) using the transform — keeps BSpline interpolation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tp_file = out_dir / "tp_img.txt"
    sitk.WriteParameterFile(last_parameter_map(tmap), str(tp_file))
    tp_text = tp_file.read_text(encoding="utf-8")
    tp_text = re.sub(
        r'\(InitialTransformParametersFileName\s+"[^"]*"\)',
        '(InitialTransformParametersFileName "NoInitialTransform")', tp_text)
    tp_file.write_text(tp_text, encoding="utf-8")
    tfx = sitk.TransformixImageFilter()
    tfx.SetMovingImage(moving_img)
    tfx.SetTransformParameterMap(sitk.ReadParameterFile(str(tp_file)))
    tfx.SetOutputDirectory(str(out_dir))
    tfx.LogToConsoleOff()
    tfx.LogToFileOff()
    tfx.Execute()
    return tfx.GetResultImage()


# ============================================================
# Settings — change TEST_PATIENT to run a different patient
# ============================================================
ATLAS_SIZE = 64
TEST_PATIENT = 96
PRESELECTION_SIZE = 5
FUSION_SIZE = 5
BSPLINE_BLACKLIST = {34}
VISUALISATION_SLICE = 15
SAVE_DIR = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results")

# ============================================================
print("Loading data...")
t0 = time.perf_counter()

loader = ProstateLoader()
images, segmen = loader.LoadData()

atlas_images = images[0:ATLAS_SIZE]
atlas_segmen = segmen[0:ATLAS_SIZE]
fixed_img = images[TEST_PATIENT]
gt_mask = segmen[TEST_PATIENT]

t_load = time.perf_counter()
print(f"Loaded in {t_load - t0:.1f}s.  Atlas: {ATLAS_SIZE}  Patient: {TEST_PATIENT}")

# ============================================================
# Affine registration
# ============================================================
print(f"\nAffine registering {ATLAS_SIZE} atlases...")
t_aff_start = time.perf_counter()

elx = sitk.ElastixImageFilter()
elx.SetFixedImage(fixed_img)
pm_affine = sitk.ReadParameterFile(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results\affine.txt")
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
print(f"Affine done in {t_aff_end - t_aff_start:.1f}s  ({(t_aff_end - t_aff_start)/ATLAS_SIZE:.1f}s per atlas)")

# ============================================================
# Preselect top 5
# ============================================================
ranked = sorted(range(ATLAS_SIZE), key=lambda k: metrics[k], reverse=True)
top_indices = ranked[:PRESELECTION_SIZE]
top_metrics = [metrics[i] for i in top_indices]
top_tmaps = [transforms[i] for i in top_indices]

print(f"\nTop {PRESELECTION_SIZE} atlases: {top_indices}")
print(f"Top metrics: {[f'{m:.4f}' for m in top_metrics]}")

# ============================================================
# BSpline refinement on top 5
# ============================================================
print(f"\nBSpline on top {PRESELECTION_SIZE}...")
t_bs_start = time.perf_counter()

bs_metrics = []
bs_tmaps = []
used_stage = []
# Store registered atlas IMAGES for visualization
registered_images = []

with TemporaryDirectory(prefix="speedtest_") as tmp:
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
            print(f"  Atlas {atlas_idx:03d} SKIPPED (blacklisted)")
            bs_metrics.append(None)
            bs_tmaps.append(top_tmaps[i])
            used_stage.append("AFF")
            # Warp image with affine for visualization
            img_dir = tmp_root / f"img_{atlas_idx:03d}"
            reg_img = warp_image(atlas_images[atlas_idx], top_tmaps[i], img_dir)
            registered_images.append(sitk.GetArrayFromImage(reg_img))
            continue

        try:
            pm_bs = sitk.ReadParameterFile("ParameterFiles/BSpline/bspline.txt")
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
                print(f"  Atlas {atlas_idx:03d}  AFF={top_metrics[i]:.3f} BS={bs_metric:.3f} -> BS")
                bs_metrics.append(bs_metric)
                bs_tmaps.append(elx_bs.GetTransformParameterMap())
                used_stage.append("BS")
                # Warp image with BSpline for visualization
                img_dir = tmp_root / f"img_{atlas_idx:03d}"
                reg_img = warp_image(atlas_images[atlas_idx], elx_bs.GetTransformParameterMap(), img_dir)
                registered_images.append(sitk.GetArrayFromImage(reg_img))
            else:
                print(f"  Atlas {atlas_idx:03d}  AFF={top_metrics[i]:.3f} BS={bs_metric:.3f} -> AFF")
                bs_metrics.append(bs_metric)
                bs_tmaps.append(top_tmaps[i])
                used_stage.append("AFF")
                # Warp image with affine for visualization
                img_dir = tmp_root / f"img_{atlas_idx:03d}"
                reg_img = warp_image(atlas_images[atlas_idx], top_tmaps[i], img_dir)
                registered_images.append(sitk.GetArrayFromImage(reg_img))

        except Exception as e:
            print(f"  Atlas {atlas_idx:03d}  FAIL: {str(e)[:80]}")
            bs_metrics.append(None)
            bs_tmaps.append(top_tmaps[i])
            used_stage.append("AFF")
            # Warp image with affine for visualization
            img_dir = tmp_root / f"img_{atlas_idx:03d}"
            reg_img = warp_image(atlas_images[atlas_idx], top_tmaps[i], img_dir)
            registered_images.append(sitk.GetArrayFromImage(reg_img))

    t_bs_end = time.perf_counter()
    print(f"BSpline done in {t_bs_end - t_bs_start:.1f}s")

    # ============================================================
    # Select best 5, warp masks, fuse
    # ============================================================
    print(f"\nFusion (best {FUSION_SIZE})...")
    t_fuse_start = time.perf_counter()

    final_metrics = []
    for i in range(PRESELECTION_SIZE):
        if used_stage[i] == "BS" and bs_metrics[i] is not None:
            final_metrics.append(bs_metrics[i])
        else:
            final_metrics.append(top_metrics[i])

    best5 = sorted(range(PRESELECTION_SIZE), key=lambda k: final_metrics[k], reverse=True)[:FUSION_SIZE]
    print(f"  Selected atlases: {[top_indices[i] for i in best5]}")

    warped_masks = []
    for i in best5:
        atlas_idx = top_indices[i]
        warp_dir = tmp_root / f"warp_{atlas_idx:03d}"
        warped = warp_label(atlas_segmen[atlas_idx], bs_tmaps[i], warp_dir)
        warped_masks.append(warped)

    fused_mask = vote_fusion(warped_masks)

    t_fuse_end = time.perf_counter()
    print(f"Fusion done in {t_fuse_end - t_fuse_start:.1f}s")

# ============================================================
# Metrics
# ============================================================
dice = dice_score(fused_mask, gt_mask)
jacc = jaccard_score(fused_mask, gt_mask)
hd = hausdorff_distance_mm(fused_mask, gt_mask)
rvd = relative_volume_difference(fused_mask, gt_mask)

t_total = time.perf_counter() - t0

print("\n" + "=" * 50)
print(f"  PATIENT {TEST_PATIENT} — RESULTS")
print("=" * 50)
print(f"  Dice:      {dice:.4f}")
print(f"  Jaccard:   {jacc:.4f}")
print(f"  Hausdorff: {hd:.2f} mm")
print(f"  RVD:       {rvd:.4f}")
print(f"\n  Affine:  {t_aff_end - t_aff_start:.1f}s  ({ATLAS_SIZE} atlases)")
print(f"  BSpline: {t_bs_end - t_bs_start:.1f}s  ({PRESELECTION_SIZE} atlases)")
print(f"  Fusion:  {t_fuse_end - t_fuse_start:.1f}s")
print(f"  TOTAL:   {t_total:.1f}s  ({t_total/60:.1f} min)")
n_bs = sum(1 for s in used_stage if s == "BS")
print(f"  BSpline used: {n_bs}/{PRESELECTION_SIZE}")

# ============================================================
# Save figure: test image + 5 atlas registered images (like the reference)
# ============================================================
SAVE_DIR.mkdir(parents=True, exist_ok=True)

fixed_np = sitk.GetArrayFromImage(fixed_img)
fused_np = sitk.GetArrayFromImage(binarize(fused_mask))
gt_np = sitk.GetArrayFromImage(binarize(gt_mask))

# Pick best slice (most prostate area)
gt_slices = gt_np.sum(axis=(1, 2))
sl = int(np.argmax(gt_slices))
print(f"\n  Visualization slice: {sl}")

# Layout: 2 rows x 3 cols = test image + 5 atlases
n_cols = 3
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5.5 * n_rows), dpi=120)

fig.suptitle(
    f"Patient {TEST_PATIENT} — Top {PRESELECTION_SIZE} atlases, best {FUSION_SIZE} selected (by metric)",
    fontsize=15, y=0.98)

# First panel: test image
ax = axes[0, 0]
ax.imshow(fixed_np[sl], cmap="gray")
ax.set_title("Test image", fontsize=12, pad=8)
ax.set_xticks([]); ax.set_yticks([])

# 5 atlas panels
positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

for i, (row, col) in enumerate(positions):
    ax = axes[row, col]
    atlas_idx = top_indices[i]
    stage = used_stage[i]
    aff_m = top_metrics[i]
    bs_m = bs_metrics[i]

    # Show registered atlas image
    ax.imshow(registered_images[i][sl], cmap="gray")

    # Determine if this atlas was selected for fusion
    is_selected = i in best5

    # Build title
    if bs_m is not None:
        bs_str = f"{bs_m:.3f}"
    else:
        bs_str = "fail"

    if stage == "BS" and is_selected:
        title_color = "green"
        status = "BS | SELECTED"
    elif stage == "AFF" and is_selected:
        title_color = "darkorange"
        status = "AFF | SELECTED"
    elif stage == "BS":
        title_color = "gray"
        status = "BS | not used"
    else:
        title_color = "gray"
        status = "AFF | not used"

    ax.set_title(
        f"Atlas {atlas_idx:03d}\nAffine: {aff_m:.3f} | BS: {bs_str}\n{status}",
        fontsize=10, pad=6, color=title_color, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.08, hspace=0.25)

fig_path = SAVE_DIR / f"patient_{TEST_PATIENT:03d}_atlas_detail.png"
fig.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSaved: {fig_path}")

print(f"\nDone! Total time: {time.perf_counter() - t0:.1f}s")
