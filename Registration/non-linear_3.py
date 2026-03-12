"""
Affine-only baseline: 50 affine registrations -> top 5 by metric -> fusion.
No BSpline. Runs on 20 test patients and saves results.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import time
import math
import gc
import json
import numpy as np

from prostateLoader import ProstateLoader
from utils import final_metric_from_elastix_log

import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Output directory
# ============================================================
OUTPUT_DIR = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results\affine")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================
def last_parameter_map(tmap):
    try:
        return tmap[-1]
    except Exception:
        return tmap


def binarize(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def dice_score(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    return 1.0 if den == 0 else (2.0 * inter / den)


def jaccard_score(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    union = int(((p | g) > 0).sum())
    return 1.0 if union == 0 else (inter / union)


def relative_volume_difference(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    vp = float(p.sum())
    vg = float(g.sum())
    if vg == 0:
        return 0.0 if vp == 0 else np.inf
    return (vp - vg) / vg


def hausdorff_distance_mm(pred: sitk.Image, gt: sitk.Image) -> float:
    pred_b = binarize(pred)
    gt_b = binarize(gt)
    pred_sum = int(sitk.GetArrayFromImage(pred_b).sum())
    gt_sum = int(sitk.GetArrayFromImage(gt_b).sum())
    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return np.inf
    hd = sitk.HausdorffDistanceImageFilter()
    hd.Execute(pred_b, gt_b)
    return float(hd.GetHausdorffDistance())


def vote_fusion(masks, vote_threshold=None):
    if len(masks) == 0:
        raise ValueError("No masks provided for fusion.")
    arrs = [sitk.GetArrayFromImage(binarize(m)).astype(np.uint8) for m in masks]
    stack = np.stack(arrs, axis=0)
    votes = stack.sum(axis=0)
    if vote_threshold is None:
        vote_threshold = math.ceil(len(masks) / 2)
    fused = (votes >= vote_threshold).astype(np.uint8)
    fused_img = sitk.GetImageFromArray(fused)
    fused_img.CopyInformation(masks[0])
    return fused_img


def write_transform_parameter_file_from_tmap(tmap, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteParameterFile(last_parameter_map(tmap), str(out_path))
    return out_path


def warp_label_with_transformix(label_img: sitk.Image, transform_param_file: Path, out_dir: Path) -> sitk.Image:
    out_dir.mkdir(parents=True, exist_ok=True)
    tp_text = transform_param_file.read_text(encoding="utf-8")
    if '(ResampleInterpolator "FinalBSplineInterpolator")' in tp_text:
        tp_text = tp_text.replace(
            '(ResampleInterpolator "FinalBSplineInterpolator")',
            '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
        )
    elif '(ResampleInterpolator "FinalLinearInterpolator")' in tp_text:
        tp_text = tp_text.replace(
            '(ResampleInterpolator "FinalLinearInterpolator")',
            '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
        )
    tp_nn = out_dir / "transform_NN.txt"
    tp_nn.write_text(tp_text, encoding="utf-8")
    tfx = sitk.TransformixImageFilter()
    tfx.SetMovingImage(label_img)
    tfx.SetTransformParameterMap(sitk.ReadParameterFile(str(tp_nn)))
    tfx.SetOutputDirectory(str(out_dir))
    tfx.LogToConsoleOff()
    tfx.LogToFileOff()
    tfx.Execute()
    warped = tfx.GetResultImage()
    warped = sitk.Cast(warped > 0.5, sitk.sitkUInt8)
    return warped


# ============================================================
# Settings
# ============================================================
ATLAS_SIZE = 50
PRESELECTION_SIZE = 10
FUSION_SIZE = 5
VISUALISATION_SLICE = 15
VOTE_THRESHOLD = None

# ============================================================
# Load data (once)
# ============================================================
t_global_start = time.perf_counter()

loader = ProstateLoader()
images, segmen = loader.LoadData()

atlas_images = images[0:ATLAS_SIZE]
atlas_segmen = segmen[0:ATLAS_SIZE]

test_images = images[ATLAS_SIZE:]
test_segmen = segmen[ATLAS_SIZE:]
NUM_TEST_PATIENTS = len(test_images)

t_load = time.perf_counter()
print(f"Loaded {len(images)} volumes.  Atlas: {ATLAS_SIZE}  Test: {NUM_TEST_PATIENTS}")
print(f"Running AFFINE-ONLY on ALL {NUM_TEST_PATIENTS} test patients (patients {ATLAS_SIZE}..{ATLAS_SIZE + NUM_TEST_PATIENTS - 1})")
print(f"Results will be saved to: {OUTPUT_DIR}\n")

# ============================================================
# Loop over patients
# ============================================================
all_results = []

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.titlesize": 15
})

for pat_idx in range(NUM_TEST_PATIENTS):
    patient_number = ATLAS_SIZE + pat_idx

    # --- Resume: skip if already saved ---
    fig1_check = OUTPUT_DIR / f"patient_{patient_number:03d}_atlases.png"
    fig2_check = OUTPUT_DIR / f"patient_{patient_number:03d}_segmentation.png"
    if fig1_check.exists() and fig2_check.exists():
        print(f"PATIENT {pat_idx+1}/{NUM_TEST_PATIENTS} (patient {patient_number}) — SKIPPING (already saved)")
        continue

    print("=" * 60)
    print(f"PATIENT {pat_idx+1}/{NUM_TEST_PATIENTS}  —  test index {pat_idx}  (patient {patient_number})")
    print("=" * 60)

    t0 = time.perf_counter()

    fixed_img = test_images[pat_idx]
    gt_mask = test_segmen[pat_idx]

    # ----------------------------------------------------------
    # Affine registration for all 50 atlases
    # ----------------------------------------------------------
    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed_img)

    pm_affine = sitk.ReadParameterFile("ParameterFiles/Affine/affine.txt")
    elx.SetParameterMap(pm_affine)
    elx.LogToConsoleOff()
    elx.LogToFileOn()

    metrics = []
    transforms = []
    reg_results = []

    for idx, moving_img in enumerate(atlas_images):
        elx.SetMovingImage(moving_img)
        elx.Execute()

        registered_img = elx.GetResultImage()
        tmap = elx.GetTransformParameterMap()

        metric = final_metric_from_elastix_log()
        metrics.append(metric)
        reg_results.append(registered_img)
        transforms.append(tmap)

    t1 = time.perf_counter()

    # ----------------------------------------------------------
    # Preselection of top K based on affine metric
    # ----------------------------------------------------------
    results = list(zip(range(len(atlas_images)), metrics, transforms,
                       atlas_images, atlas_segmen, reg_results))
    results_sorted = sorted(results, key=lambda t: t[1], reverse=True)
    top_results = results_sorted[:PRESELECTION_SIZE]

    top_indices = [r[0] for r in top_results]
    top_metrics = [r[1] for r in top_results]
    top_tmaps   = [r[2] for r in top_results]
    top_segmen_sel = [r[4] for r in top_results]
    top_reg     = [r[5] for r in top_results]

    # Select best FUSION_SIZE by affine metric (already sorted)
    selected_indices = list(range(FUSION_SIZE))

    print(f"  Top {FUSION_SIZE} selected for fusion (by affine metric):")
    for i in selected_indices:
        print(f"    Atlas {top_indices[i]:03d} | affine metric = {top_metrics[i]:.4f}")

    t2 = time.perf_counter()

    # ----------------------------------------------------------
    # Warp selected atlas masks with affine transform
    # ----------------------------------------------------------
    with TemporaryDirectory(prefix="affine_warp_") as tmp:
        tmp_root = Path(tmp)
        warped_masks = []

        for i in selected_indices:
            atlas_idx = top_indices[i]
            work_dir = tmp_root / f"atlas_{atlas_idx:03d}"

            tp_file = write_transform_parameter_file_from_tmap(
                top_tmaps[i],
                work_dir / "affine_tp.txt"
            )

            # Fix InitialTransformParametersFileName
            tp_text = tp_file.read_text(encoding="utf-8")
            tp_text = re.sub(
                r'\(InitialTransformParametersFileName\s+"[^"]*"\)',
                '(InitialTransformParametersFileName "NoInitialTransform")',
                tp_text,
            )
            tp_file.write_text(tp_text, encoding="utf-8")

            warped_mask = warp_label_with_transformix(
                label_img=top_segmen_sel[i],
                transform_param_file=tp_file,
                out_dir=work_dir / "mask_warp",
            )
            warped_masks.append(warped_mask)

        # ------------------------------------------------------
        # Fuse and evaluate
        # ------------------------------------------------------
        fused_mask = vote_fusion(warped_masks, vote_threshold=VOTE_THRESHOLD)

        dice = dice_score(fused_mask, gt_mask)
        jacc = jaccard_score(fused_mask, gt_mask)
        hd   = hausdorff_distance_mm(fused_mask, gt_mask)
        rvd  = relative_volume_difference(fused_mask, gt_mask)

        t3 = time.perf_counter()

        # ------------------------------------------------------
        # Save figure 1: top 10 atlases (showing all, marking selected)
        # ------------------------------------------------------
        n_panels = 1 + PRESELECTION_SIZE
        n_cols = 4
        n_rows = math.ceil(n_panels / n_cols)

        fig1, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), dpi=120)
        axes = np.array(axes).reshape(-1)

        fig1.suptitle(
            f"Patient {patient_number} — Top {PRESELECTION_SIZE} atlases, "
            f"best {FUSION_SIZE} selected (AFFINE ONLY)",
            fontsize=14, y=0.98
        )

        axes[0].imshow(fixed_img[:, :, VISUALISATION_SLICE], cmap="gray")
        axes[0].set_title("Test image", fontsize=12, pad=8)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        for i in range(PRESELECTION_SIZE):
            ax = axes[i + 1]
            ax.imshow(top_reg[i][:, :, VISUALISATION_SLICE], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])

            is_selected = i in selected_indices
            status = "SELECTED" if is_selected else "not used"
            title_txt = (
                f"Atlas {top_indices[i]:03d}\n"
                f"Affine: {top_metrics[i]:.3f}\n"
                f"{status}"
            )
            color = "green" if is_selected else "gray"
            ax.set_title(title_txt, fontsize=10, pad=8, color=color)

        for k in range(n_panels, len(axes)):
            axes[k].axis("off")

        fig1.subplots_adjust(left=0.03, right=0.99, bottom=0.05, top=0.90,
                             wspace=0.08, hspace=0.35)

        fig1_path = OUTPUT_DIR / f"patient_{patient_number:03d}_atlases.png"
        fig1.savefig(fig1_path, bbox_inches="tight")
        plt.close(fig1)

        # ------------------------------------------------------
        # Save figure 2: segmentation result
        # ------------------------------------------------------
        fixed_np = sitk.GetArrayFromImage(fixed_img)
        fused_np = sitk.GetArrayFromImage(binarize(fused_mask))
        gt_np    = sitk.GetArrayFromImage(binarize(gt_mask))

        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.8), dpi=120)
        fig2.suptitle(
            f"Patient {patient_number} — Segmentation (AFFINE ONLY)\n"
            f"Dice: {dice:.3f} | Jaccard: {jacc:.3f} | "
            f"Hausdorff: {hd:.3f} mm | RVD: {rvd:.3f}",
            fontsize=14, y=0.97
        )

        axes2[0].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
        axes2[0].set_title("Test image", fontsize=13, pad=8)
        axes2[0].set_xticks([]); axes2[0].set_yticks([])

        axes2[1].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
        axes2[1].imshow(
            np.ma.masked_where(fused_np[VISUALISATION_SLICE] == 0,
                               fused_np[VISUALISATION_SLICE]),
            cmap="Reds", alpha=0.4,
        )
        axes2[1].set_title("Fused prediction", fontsize=13, pad=8)
        axes2[1].set_xticks([]); axes2[1].set_yticks([])

        axes2[2].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
        axes2[2].imshow(
            np.ma.masked_where(gt_np[VISUALISATION_SLICE] == 0,
                               gt_np[VISUALISATION_SLICE]),
            cmap="Reds", alpha=0.4,
        )
        axes2[2].set_title("Ground truth mask", fontsize=13, pad=8)
        axes2[2].set_xticks([]); axes2[2].set_yticks([])

        fig2.subplots_adjust(left=0.03, right=0.99, bottom=0.06, top=0.80,
                             wspace=0.03)

        fig2_path = OUTPUT_DIR / f"patient_{patient_number:03d}_segmentation.png"
        fig2.savefig(fig2_path, bbox_inches="tight")
        plt.close(fig2)

    # ----------------------------------------------------------
    # Store result
    # ----------------------------------------------------------
    pat_time = t3 - t0

    patient_result = {
        "patient_number": patient_number,
        "test_index": pat_idx,
        "dice": dice,
        "jaccard": jacc,
        "hausdorff": hd,
        "rvd": rvd,
        "time_s": pat_time,
        "top_indices": list(top_indices),
        "top_affine_metrics": list(top_metrics),
        "selected_for_fusion": [top_indices[i] for i in selected_indices],
    }
    all_results.append(patient_result)

    # Save per-patient JSON so the report can be rebuilt on resume
    json_path = OUTPUT_DIR / f"patient_{patient_number:03d}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(patient_result, jf, indent=2, default=str)

    print(f"  Dice={dice:.3f}  Jacc={jacc:.3f}  HD={hd:.1f}mm  RVD={rvd:.3f}  "
          f"time={pat_time:.0f}s")
    print()

    # Free memory
    del metrics, transforms, reg_results, warped_masks, fused_mask, fixed_img, gt_mask
    gc.collect()


# ============================================================
# Rebuild full results from all per-patient JSONs (covers previous runs too)
# ============================================================
t_global_end = time.perf_counter()

all_results = []
for pat_idx in range(NUM_TEST_PATIENTS):
    patient_number = ATLAS_SIZE + pat_idx
    json_path = OUTPUT_DIR / f"patient_{patient_number:03d}_metrics.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as jf:
            all_results.append(json.load(jf))

print(f"\nLoaded results for {len(all_results)}/{NUM_TEST_PATIENTS} patients.")

if len(all_results) == 0:
    print("No patient results found — nothing to report.")
else:
    dices  = [r["dice"] for r in all_results]
    jaccs  = [r["jaccard"] for r in all_results]
    hds    = [r["hausdorff"] for r in all_results]
    rvds   = [r["rvd"] for r in all_results]

    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  AFFINE-ONLY REGISTRATION REPORT\n")
        f.write(f"  {len(all_results)}/{NUM_TEST_PATIENTS} test patients  |  "
                f"Top {PRESELECTION_SIZE} affine -> Best {FUSION_SIZE} for fusion\n")
        f.write("  NO BSpline — affine registration only\n")
        f.write("=" * 70 + "\n\n")

        f.write("SETTINGS\n")
        f.write(f"  Atlas size            : {ATLAS_SIZE}\n")
        f.write(f"  Preselection size     : {PRESELECTION_SIZE}\n")
        f.write(f"  Fusion size           : {FUSION_SIZE}\n")
        f.write(f"  Vote threshold        : {VOTE_THRESHOLD}\n")
        f.write(f"  Affine param file     : ParameterFiles/Affine/affine.txt\n")
        f.write(f"  Total time            : {(t_global_end - t_global_start)/60:.1f} min\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'Patient':>8}  {'Dice':>6}  {'Jacc':>6}  {'HD(mm)':>7}  "
                f"{'RVD':>7}  {'Time':>6}\n")
        f.write("-" * 70 + "\n")

        for r in all_results:
            f.write(f"{r['patient_number']:>8}  "
                    f"{r['dice']:>6.3f}  "
                    f"{r['jaccard']:>6.3f}  "
                    f"{r['hausdorff']:>7.1f}  "
                    f"{r['rvd']:>7.3f}  "
                    f"{r['time_s']:>5.0f}s\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'MEAN':>8}  {np.mean(dices):>6.3f}  {np.mean(jaccs):>6.3f}  "
                f"{np.mean(hds):>7.1f}  {np.mean(rvds):>7.3f}\n")
        f.write(f"{'STD':>8}  {np.std(dices):>6.3f}  {np.std(jaccs):>6.3f}  "
                f"{np.std(hds):>7.1f}  {np.std(rvds):>7.3f}\n")
        f.write(f"{'MEDIAN':>8}  {np.median(dices):>6.3f}  {np.median(jaccs):>6.3f}  "
                f"{np.median(hds):>7.1f}  {np.median(rvds):>7.3f}\n\n")

        for r in all_results:
            f.write(f"Patient {r['patient_number']}: "
                    f"Dice={r['dice']:.4f}  Jacc={r['jaccard']:.4f}  "
                    f"HD={r['hausdorff']:.2f}mm  RVD={r['rvd']:.4f}  "
                    f"Fused={r['selected_for_fusion']}\n")

    print("=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"Total time: {(t_global_end - t_global_start)/60:.1f} min")
    print(f"\nOverall metrics across {len(all_results)} patients (AFFINE ONLY):")
    print(f"  Dice    : {np.mean(dices):.3f} +/- {np.std(dices):.3f}")
    print(f"  Jaccard : {np.mean(jaccs):.3f} +/- {np.std(jaccs):.3f}")
    print(f"  Hausdorf: {np.mean(hds):.1f} +/- {np.std(hds):.1f} mm")
    print(f"  RVD     : {np.mean(rvds):.3f} +/- {np.std(rvds):.3f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
