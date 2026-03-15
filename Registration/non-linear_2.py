"""
Batch pipeline: 64 affine -> top 5 -> BSpline -> best 5 -> fusion
Loops through all test patients (64-138).
Skips patients with existing output PNGs (resume capable).
Saves segmentation figures + metrics JSON to top5_fast/
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
# Settings
# ============================================================
ATLAS_SIZE = 64
TEST_START = 64           # first test patient index
PRESELECTION_SIZE = 5
FUSION_SIZE = 5
BSPLINE_BLACKLIST = {34}
VISUALISATION_SLICE = 15
SAVE_DIR = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results\top5_fast")

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

    # --- Affine ---
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

    # --- Preselect top 5 ---
    ranked = sorted(range(ATLAS_SIZE), key=lambda k: metrics[k], reverse=True)
    top_indices = ranked[:PRESELECTION_SIZE]
    top_metrics = [metrics[i] for i in top_indices]
    top_tmaps = [transforms[i] for i in top_indices]

    print(f"  Top {PRESELECTION_SIZE} atlases: {top_indices}")

    # --- BSpline ---
    print(f"  BSpline on top {PRESELECTION_SIZE}...")
    t_bs_start = time.perf_counter()

    bs_metrics = []
    bs_tmaps = []
    used_stage = []

    with TemporaryDirectory(prefix="batch_") as tmp:
        tmp_root = Path(tmp)

        for i, atlas_idx in enumerate(top_indices):
            work_dir = tmp_root / f"atlas_{atlas_idx:03d}"
            work_dir.mkdir(parents=True, exist_ok=True)

            aff_file = work_dir / "affine_tp.txt"
            sitk.WriteParameterFile(last_parameter_map(top_tmaps[i]), str(aff_file))
            tp_text = aff_file.read_text(encoding="utf-8")
            tp_text = re.sub(
                r'\(InitialTransformParametersFileName\s+"[^"]*"\)',
                '(InitialTransformParametersFileName "NoInitialTransform")', tp_text)
            aff_file.write_text(tp_text, encoding="utf-8")

            if atlas_idx in BSPLINE_BLACKLIST:
                print(f"    Atlas {atlas_idx:03d} SKIPPED (blacklisted)")
                bs_metrics.append(None)
                bs_tmaps.append(top_tmaps[i])
                used_stage.append("AFF")
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
                    print(f"    Atlas {atlas_idx:03d}  AFF={top_metrics[i]:.3f} BS={bs_metric:.3f} -> BS")
                    bs_metrics.append(bs_metric)
                    bs_tmaps.append(elx_bs.GetTransformParameterMap())
                    used_stage.append("BS")
                else:
                    print(f"    Atlas {atlas_idx:03d}  AFF={top_metrics[i]:.3f} BS={bs_metric:.3f} -> AFF")
                    bs_metrics.append(bs_metric)
                    bs_tmaps.append(top_tmaps[i])
                    used_stage.append("AFF")

            except Exception as e:
                print(f"    Atlas {atlas_idx:03d}  FAIL: {str(e)[:80]}")
                bs_metrics.append(None)
                bs_tmaps.append(top_tmaps[i])
                used_stage.append("AFF")

        t_bs_end = time.perf_counter()
        print(f"  BSpline done in {t_bs_end - t_bs_start:.1f}s")

        # --- Fusion ---
        print(f"  Fusion (best {FUSION_SIZE})...")
        t_fuse_start = time.perf_counter()

        final_metrics = []
        for i in range(PRESELECTION_SIZE):
            if used_stage[i] == "BS" and bs_metrics[i] is not None:
                final_metrics.append(bs_metrics[i])
            else:
                final_metrics.append(top_metrics[i])

        best5 = sorted(range(PRESELECTION_SIZE), key=lambda k: final_metrics[k], reverse=True)[:FUSION_SIZE]
        print(f"    Selected atlases: {[top_indices[i] for i in best5]}")

        warped_masks = []
        for i in best5:
            atlas_idx = top_indices[i]
            warp_dir = tmp_root / f"warp_{atlas_idx:03d}"
            warped = warp_label(atlas_segmen[atlas_idx], bs_tmaps[i], warp_dir)
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
    n_bs = sum(1 for s in used_stage if s == "BS")

    result = {
        "patient": test_idx,
        "dice": round(dice, 4),
        "jaccard": round(jacc, 4),
        "hausdorff": round(hd, 4),
        "rvd": round(rvd, 4),
        "time_s": round(t_total, 1),
        "bspline_used": n_bs,
        "atlases_used": [top_indices[i] for i in best5]
    }
    all_results.append(result)

    print(f"\n  Dice: {dice:.4f}  Jaccard: {jacc:.4f}  HD: {hd:.2f}mm  RVD: {rvd:.4f}")
    print(f"  Time: {t_total:.1f}s ({t_total/60:.1f}min)  BSpline: {n_bs}/{PRESELECTION_SIZE}")

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
        f"Patient {test_idx} — {ATLAS_SIZE} atlases, top {PRESELECTION_SIZE} -> BS -> best {FUSION_SIZE}\n"
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
