from pathlib import Path
from tempfile import TemporaryDirectory
import re
import time
import math
import numpy as np

from prostateLoader import ProstateLoader
from utils import final_metric_from_elastix_log

import gc
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")          # no GUI — save figures only
import matplotlib.pyplot as plt


# ============================================================
# Output directory
# ============================================================
OUTPUT_DIR = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions (unchanged)
# ============================================================
def final_metric_from_log(log_path: Path) -> float:
    if not log_path.exists():
        raise FileNotFoundError(f"No elastix.log found at {log_path}")

    pat = re.compile(r"Final metric value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    metric = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                metric = float(m.group(1))

    if metric is None:
        raise RuntimeError(f"Could not find 'Final metric value' in {log_path}")

    return -metric


def last_parameter_map(tmap):
    try:
        return tmap[-1]
    except Exception:
        return tmap


def latest_tp(out_dir: Path) -> Path:
    tps = sorted(out_dir.glob("TransformParameters.*.txt"))
    if not tps:
        raise FileNotFoundError(f"No TransformParameters.*.txt found in: {out_dir}")
    return tps[-1]


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


def run_bspline_from_affine(fixed_img, moving_img, affine_tmap, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)

    affine_tp_file = work_dir / "affine_tp.txt"
    sitk.WriteParameterFile(last_parameter_map(affine_tmap), str(affine_tp_file))

    tp_text = affine_tp_file.read_text(encoding="utf-8")
    tp_text = re.sub(
        r'\(InitialTransformParametersFileName\s+"[^"]*"\)',
        '(InitialTransformParametersFileName "NoInitialTransform")',
        tp_text,
    )
    affine_tp_file.write_text(tp_text, encoding="utf-8")

    pm_bspline = sitk.ReadParameterFile("ParameterFiles/BSpline/bspline.txt")
    pm_bspline["InitialTransformParameterFileName"] = [str(affine_tp_file).replace("\\", "/")]
    pm_bspline["HowToCombineTransforms"] = ["Compose"]

    elx_bspline = sitk.ElastixImageFilter()
    elx_bspline.SetFixedImage(fixed_img)
    elx_bspline.SetMovingImage(moving_img)
    elx_bspline.SetParameterMap(pm_bspline)
    elx_bspline.SetOutputDirectory(str(work_dir))
    elx_bspline.LogToConsoleOff()
    elx_bspline.LogToFileOn()
    elx_bspline.Execute()

    registered_img = elx_bspline.GetResultImage()
    bs_metric = final_metric_from_log(work_dir / "elastix.log")
    bs_tp_file = latest_tp(work_dir)

    return registered_img, bs_metric, bs_tp_file


# ============================================================
# Settings
# ============================================================
ATLAS_SIZE = 50
PRESELECTION_SIZE = 10
FUSION_SIZE = 5
NUM_TEST_PATIENTS = 20
VISUALISATION_SLICE = 15
VOTE_THRESHOLD = None

# Atlases that cause segfaults during BSpline — skip BSpline for these
BSPLINE_BLACKLIST = {34}

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

t_load = time.perf_counter()
print(f"Loaded {len(images)} volumes.  Atlas: {ATLAS_SIZE}  Test: {len(test_images)}")
print(f"Running on {NUM_TEST_PATIENTS} test patients (indices 0..{NUM_TEST_PATIENTS-1})")
print(f"Results will be saved to: {OUTPUT_DIR}\n")

# ============================================================
# Collect results across all patients
# ============================================================
all_results = []   # list of dicts, one per patient

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.titlesize": 15
})

for pat_idx in range(NUM_TEST_PATIENTS):
    patient_number = ATLAS_SIZE + pat_idx

    # --- Resume: skip if both images already saved ---
    fig1_path_check = OUTPUT_DIR / f"patient_{patient_number:03d}_atlases.png"
    fig2_path_check = OUTPUT_DIR / f"patient_{patient_number:03d}_segmentation.png"
    if fig1_path_check.exists() and fig2_path_check.exists():
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
    top_images  = [r[3] for r in top_results]
    top_segmen_sel = [r[4] for r in top_results]
    top_reg     = [r[5] for r in top_results]

    t2 = time.perf_counter()

    # ----------------------------------------------------------
    # BSpline refinement on top K
    # ----------------------------------------------------------
    final_images = []
    final_transform_files = []
    bs_metrics = []
    used_stage = []
    bs_errors = []        # collect error messages

    with TemporaryDirectory(prefix="bspline_topk_") as tmp:
        tmp_root = Path(tmp)

        for i, moving_img in enumerate(top_images):
            atlas_idx = top_indices[i]
            work_dir = tmp_root / f"atlas_{atlas_idx:03d}"

            affine_tp_file = write_transform_parameter_file_from_tmap(
                top_tmaps[i],
                work_dir / "affine_only_tp.txt"
            )

            # Skip BSpline for blacklisted atlases (known segfault)
            if atlas_idx in BSPLINE_BLACKLIST:
                print(f"  BSpline atlas {atlas_idx:03d} ... SKIPPED (blacklisted)")
                final_images.append(top_reg[i])
                final_transform_files.append(affine_tp_file)
                bs_metrics.append(None)
                used_stage.append("AFF")
                bs_errors.append("Blacklisted (known segfault)")
                continue

            try:
                print(f"  BSpline atlas {atlas_idx:03d} ...", end=" ")
                bspline_img, bs_metric, bs_tp_file = run_bspline_from_affine(
                    fixed_img=fixed_img,
                    moving_img=moving_img,
                    affine_tmap=top_tmaps[i],
                    work_dir=work_dir,
                )

                if bs_metric > top_metrics[i]:
                    print(f"AFF={top_metrics[i]:.3f} BS={bs_metric:.3f} -> BS")
                    final_images.append(bspline_img)
                    final_transform_files.append(bs_tp_file)
                    bs_metrics.append(bs_metric)
                    used_stage.append("BS")
                    bs_errors.append(None)
                else:
                    print(f"AFF={top_metrics[i]:.3f} BS={bs_metric:.3f} -> AFF (BS worse)")
                    final_images.append(top_reg[i])
                    final_transform_files.append(affine_tp_file)
                    bs_metrics.append(bs_metric)
                    used_stage.append("AFF")
                    bs_errors.append("BSpline worse than affine")

            except Exception as e:
                err_msg = str(e).split("\n")[0][:120]
                log_snippet = ""
                crash_log = work_dir / "elastix.log"
                if crash_log.exists():
                    lines = crash_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                    for ln in reversed(lines):
                        if "Description:" in ln:
                            log_snippet = ln.strip()
                            break

                print(f"FAIL ({log_snippet or err_msg[:60]})")

                final_images.append(top_reg[i])
                final_transform_files.append(affine_tp_file)
                bs_metrics.append(None)
                used_stage.append("AFF")
                bs_errors.append(log_snippet or err_msg)

        t3 = time.perf_counter()

        # ------------------------------------------------------
        # Select best FUSION_SIZE by final metric
        # ------------------------------------------------------
        final_metric_values = []
        for i in range(len(top_indices)):
            if used_stage[i] == "BS" and bs_metrics[i] is not None:
                final_metric_values.append(bs_metrics[i])
            else:
                final_metric_values.append(top_metrics[i])

        ranked = sorted(range(len(top_indices)),
                        key=lambda k: final_metric_values[k], reverse=True)
        selected_indices = ranked[:FUSION_SIZE]

        # ------------------------------------------------------
        # Warp masks & fuse
        # ------------------------------------------------------
        warped_masks = []

        for i in selected_indices:
            atlas_idx = top_indices[i]
            warp_dir = tmp_root / f"atlas_{atlas_idx:03d}" / "mask_warp"

            warped_mask = warp_label_with_transformix(
                label_img=top_segmen_sel[i],
                transform_param_file=final_transform_files[i],
                out_dir=warp_dir,
            )
            warped_masks.append(warped_mask)

        fused_mask = vote_fusion(warped_masks, vote_threshold=VOTE_THRESHOLD)

        dice = dice_score(fused_mask, gt_mask)
        jacc = jaccard_score(fused_mask, gt_mask)
        hd   = hausdorff_distance_mm(fused_mask, gt_mask)
        rvd  = relative_volume_difference(fused_mask, gt_mask)

        t4 = time.perf_counter()

        # ------------------------------------------------------
        # Save figure 1: atlas comparison
        # ------------------------------------------------------
        n_panels = 1 + len(final_images)
        n_cols = 4
        n_rows = math.ceil(n_panels / n_cols)

        fig1, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows), dpi=120)
        axes = np.array(axes).reshape(-1)

        fig1.suptitle(
            f"Patient {patient_number} — Top {PRESELECTION_SIZE} atlases, "
            f"best {FUSION_SIZE} selected (by metric)",
            fontsize=14, y=0.98
        )

        axes[0].imshow(fixed_img[:, :, VISUALISATION_SLICE], cmap="gray")
        axes[0].set_title("Test image", fontsize=12, pad=8)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        for i, img in enumerate(final_images):
            ax = axes[i + 1]
            ax.imshow(img[:, :, VISUALISATION_SLICE], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])

            is_selected = i in selected_indices
            bs_str = "fail" if bs_metrics[i] is None else f"{bs_metrics[i]:.3f}"
            status = "SELECTED" if is_selected else "not used"
            title_txt = (
                f"Atlas {top_indices[i]:03d}\n"
                f"Affine: {top_metrics[i]:.3f} | BS: {bs_str}\n"
                f"{used_stage[i]} | {status}"
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
            f"Patient {patient_number} — Segmentation result\n"
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
    # Store per-patient results
    # ----------------------------------------------------------
    pat_time = t4 - t0
    n_bs_used = sum(1 for s in used_stage if s == "BS")
    n_bs_fail = sum(1 for b in bs_metrics if b is None)
    n_bs_worse = sum(1 for i, s in enumerate(used_stage)
                     if s == "AFF" and bs_metrics[i] is not None)

    patient_result = {
        "patient_number": patient_number,
        "test_index": pat_idx,
        "dice": dice,
        "jaccard": jacc,
        "hausdorff": hd,
        "rvd": rvd,
        "time_s": pat_time,
        "n_bs_used": n_bs_used,
        "n_bs_fail": n_bs_fail,
        "n_bs_worse": n_bs_worse,
        "top_indices": list(top_indices),
        "top_affine_metrics": list(top_metrics),
        "bs_metrics": list(bs_metrics),
        "used_stage": list(used_stage),
        "selected_for_fusion": [top_indices[i] for i in selected_indices],
        "bs_errors": list(bs_errors),
    }
    all_results.append(patient_result)

    print(f"  Dice={dice:.3f}  Jacc={jacc:.3f}  HD={hd:.1f}mm  RVD={rvd:.3f}  "
          f"BS used={n_bs_used}/10  time={pat_time:.0f}s")
    print()

    # Free memory before next patient
    del metrics, transforms, reg_results, final_images, final_transform_files
    del warped_masks, fused_mask, fixed_img, gt_mask
    gc.collect()


# ============================================================
# Write detailed report
# ============================================================
t_global_end = time.perf_counter()

report_path = OUTPUT_DIR / "report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("  NON-LINEAR REGISTRATION REPORT\n")
    f.write(f"  {NUM_TEST_PATIENTS} test patients  |  "
            f"Top {PRESELECTION_SIZE} affine -> Best {FUSION_SIZE} for fusion\n")
    f.write("=" * 70 + "\n\n")

    # Settings
    f.write("SETTINGS\n")
    f.write(f"  Atlas size            : {ATLAS_SIZE}\n")
    f.write(f"  Preselection size     : {PRESELECTION_SIZE}\n")
    f.write(f"  Fusion size           : {FUSION_SIZE}\n")
    f.write(f"  Vote threshold        : {VOTE_THRESHOLD}\n")
    f.write(f"  Visualisation slice   : {VISUALISATION_SLICE}\n")
    f.write(f"  BSpline param file    : ParameterFiles/BSpline/bspline.txt\n")
    f.write(f"  Affine param file     : ParameterFiles/Affine/affine.txt\n")
    f.write(f"  Total time            : {(t_global_end - t_global_start)/60:.1f} min\n\n")

    # Per-patient table
    f.write("-" * 70 + "\n")
    f.write(f"{'Patient':>8}  {'Dice':>6}  {'Jacc':>6}  {'HD(mm)':>7}  "
            f"{'RVD':>7}  {'BS ok':>5}  {'BS fail':>7}  {'Time':>6}\n")
    f.write("-" * 70 + "\n")

    dices, jaccs, hds, rvds = [], [], [], []
    for r in all_results:
        f.write(f"{r['patient_number']:>8}  "
                f"{r['dice']:>6.3f}  "
                f"{r['jaccard']:>6.3f}  "
                f"{r['hausdorff']:>7.1f}  "
                f"{r['rvd']:>7.3f}  "
                f"{r['n_bs_used']:>5}  "
                f"{r['n_bs_fail']:>7}  "
                f"{r['time_s']:>5.0f}s\n")
        dices.append(r["dice"])
        jaccs.append(r["jaccard"])
        hds.append(r["hausdorff"])
        rvds.append(r["rvd"])

    f.write("-" * 70 + "\n")
    f.write(f"{'MEAN':>8}  "
            f"{np.mean(dices):>6.3f}  "
            f"{np.mean(jaccs):>6.3f}  "
            f"{np.mean(hds):>7.1f}  "
            f"{np.mean(rvds):>7.3f}\n")
    f.write(f"{'STD':>8}  "
            f"{np.std(dices):>6.3f}  "
            f"{np.std(jaccs):>6.3f}  "
            f"{np.std(hds):>7.1f}  "
            f"{np.std(rvds):>7.3f}\n")
    f.write(f"{'MEDIAN':>8}  "
            f"{np.median(dices):>6.3f}  "
            f"{np.median(jaccs):>6.3f}  "
            f"{np.median(hds):>7.1f}  "
            f"{np.median(rvds):>7.3f}\n")
    f.write(f"{'MIN':>8}  "
            f"{np.min(dices):>6.3f}  "
            f"{np.min(jaccs):>6.3f}  "
            f"{np.min(hds):>7.1f}  "
            f"{np.min(rvds):>7.3f}\n")
    f.write(f"{'MAX':>8}  "
            f"{np.max(dices):>6.3f}  "
            f"{np.max(jaccs):>6.3f}  "
            f"{np.max(hds):>7.1f}  "
            f"{np.max(rvds):>7.3f}\n\n")

    # Per-patient detail
    for r in all_results:
        f.write("=" * 70 + "\n")
        f.write(f"Patient {r['patient_number']} (test index {r['test_index']})\n")
        f.write(f"  Dice={r['dice']:.4f}  Jaccard={r['jaccard']:.4f}  "
                f"HD={r['hausdorff']:.2f}mm  RVD={r['rvd']:.4f}\n")
        f.write(f"  BSpline used: {r['n_bs_used']}/10  |  "
                f"BSpline failed: {r['n_bs_fail']}/10  |  "
                f"BSpline worse: {r['n_bs_worse']}/10\n")
        f.write(f"  Selected for fusion: {r['selected_for_fusion']}\n")
        f.write(f"  Time: {r['time_s']:.0f}s\n\n")

        f.write(f"  {'Atlas':>6}  {'AffMetric':>9}  {'BSMetric':>9}  "
                f"{'Stage':>5}  Error\n")
        for i in range(len(r["top_indices"])):
            bs_str = "fail" if r["bs_metrics"][i] is None else f"{r['bs_metrics'][i]:.4f}"
            err = r["bs_errors"][i] or ""
            f.write(f"  {r['top_indices'][i]:>6}  "
                    f"{r['top_affine_metrics'][i]:>9.4f}  "
                    f"{bs_str:>9}  "
                    f"{r['used_stage'][i]:>5}  "
                    f"{err}\n")
        f.write("\n")

print("=" * 60)
print("ALL DONE")
print("=" * 60)
print(f"Total time: {(t_global_end - t_global_start)/60:.1f} min")
print(f"\nOverall metrics across {NUM_TEST_PATIENTS} patients:")
print(f"  Dice    : {np.mean(dices):.3f} +/- {np.std(dices):.3f}  (range {np.min(dices):.3f} - {np.max(dices):.3f})")
print(f"  Jaccard : {np.mean(jaccs):.3f} +/- {np.std(jaccs):.3f}")
print(f"  Hausdorf: {np.mean(hds):.1f} +/- {np.std(hds):.1f} mm")
print(f"  RVD     : {np.mean(rvds):.3f} +/- {np.std(rvds):.3f}")
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  - {NUM_TEST_PATIENTS} atlas comparison images")
print(f"  - {NUM_TEST_PATIENTS} segmentation result images")
print(f"  - report.txt (detailed per-patient breakdown)")
