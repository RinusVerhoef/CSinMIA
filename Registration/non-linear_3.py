from pathlib import Path
from tempfile import TemporaryDirectory
import re
import time
import math
import numpy as np
from prostateLoader import ProstateLoader
from utils import final_metric_from_elastix_log
import SimpleITK as sitk
import matplotlib.pyplot as plt

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

    # Fix: ensure the written affine transform does not reference a
    # stale InitialTransformParameterFileName path that no longer exists.
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

# ------------------------------------------------------------
# Timer start
# ------------------------------------------------------------
t0 = time.perf_counter()
# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------
ATLAS_SIZE = 50
PRESELECTION_SIZE = 5
VISUALISATION_SLICE = 15
VOTE_THRESHOLD = None
METRIC_THRESHOLD = 0.7          # only fuse atlases with final metric >= this
# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
loader = ProstateLoader()
images, segmen = loader.LoadData()
atlas_images = images[0:ATLAS_SIZE]
atlas_segmen = segmen[0:ATLAS_SIZE]
test_images = images[ATLAS_SIZE:]
test_segmen = segmen[ATLAS_SIZE:]
t1 = time.perf_counter()
# ------------------------------------------------------------
# Select fixed image and GT mask (51st patient = test_images[0])
# ------------------------------------------------------------
fixed_img = test_images[0]
gt_mask = test_segmen[0]
# ------------------------------------------------------------
# Affine registration for all 50 atlases
# ------------------------------------------------------------
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
t2 = time.perf_counter()
# ------------------------------------------------------------
# Preselection of top 5 based on affine metric
# ------------------------------------------------------------
results = list(zip(range(len(atlas_images)), metrics, transforms, atlas_images, atlas_segmen, reg_results))
results_sorted = sorted(results, key=lambda t: t[1], reverse=True)
top_results = results_sorted[:PRESELECTION_SIZE]
top_indices = [r[0] for r in top_results]
top_metrics = [r[1] for r in top_results]
top_tmaps = [r[2] for r in top_results]
top_images = [r[3] for r in top_results]
top_segmen = [r[4] for r in top_results]
top_reg = [r[5] for r in top_results]
t3 = time.perf_counter()
# ------------------------------------------------------------
# Run BSpline only on top 5
# Keep affine if BSpline is worse or fails
# Also keep the chosen transform file for mask warping
# ------------------------------------------------------------
final_images = []
final_transform_files = []
final_metrics = []              # the metric of whichever stage was chosen
bs_metrics = []
used_stage = []
with TemporaryDirectory(prefix="bspline_top5_") as tmp:
    tmp_root = Path(tmp)
    for i, moving_img in enumerate(top_images):
        atlas_idx = top_indices[i]
        work_dir = tmp_root / f"atlas_{atlas_idx:03d}"
        affine_tp_file = write_transform_parameter_file_from_tmap(
            top_tmaps[i],
            work_dir / "affine_only_tp.txt"
        )
        try:
            print(f"  Running BSpline for atlas {atlas_idx:03d} ...")
            bspline_img, bs_metric, bs_tp_file = run_bspline_from_affine(
                fixed_img=fixed_img,
                moving_img=moving_img,
                affine_tmap=top_tmaps[i],
                work_dir=work_dir,
            )
            print(f"    Affine metric: {top_metrics[i]:.6f}  |  BSpline metric: {bs_metric:.6f}")
            if bs_metric > top_metrics[i]:
                print(f"    -> BSpline is BETTER, using BSpline")
                final_images.append(bspline_img)
                final_transform_files.append(bs_tp_file)
                final_metrics.append(bs_metric)
                bs_metrics.append(bs_metric)
                used_stage.append("BS")
            else:
                print(f"    -> BSpline is WORSE, keeping affine")
                final_images.append(top_reg[i])
                final_transform_files.append(affine_tp_file)
                final_metrics.append(top_metrics[i])
                bs_metrics.append(bs_metric)
                used_stage.append("AFF")
        except Exception as e:
            print(f"  BSpline CRASHED for atlas {atlas_idx:03d}: {e}")
            crash_log = work_dir / "elastix.log"
            if crash_log.exists():
                print(f"  --- elastix.log (last 15 lines) ---")
                lines = crash_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                for ln in lines[-15:]:
                    print(f"    {ln}")
                print(f"  --- end log ---")
            final_images.append(top_reg[i])
            final_transform_files.append(affine_tp_file)
            final_metrics.append(top_metrics[i])
            bs_metrics.append(None)
            used_stage.append("AFF")
    t4 = time.perf_counter()
    # --------------------------------------------------------
    # Filter: only keep atlases whose final metric >= threshold
    # --------------------------------------------------------
    keep = [j for j in range(len(final_metrics)) if final_metrics[j] >= METRIC_THRESHOLD]
    if len(keep) == 0:
        print(f"\n  WARNING: no atlas reached the threshold ({METRIC_THRESHOLD}).")
        print(f"  Falling back to the single best atlas.")
        best = int(np.argmax(final_metrics))
        keep = [best]

    print(f"\n  Metric threshold: {METRIC_THRESHOLD}")
    print(f"  Atlases kept for fusion: {len(keep)} / {len(final_metrics)}")
    for j in keep:
        print(f"    Atlas {top_indices[j]:03d}  metric={final_metrics[j]:.6f}  stage={used_stage[j]}")

    # --------------------------------------------------------
    # Warp only the kept atlas masks
    # --------------------------------------------------------
    warped_masks = []
    for j in keep:
        atlas_idx = top_indices[j]
        warp_dir = tmp_root / f"atlas_{atlas_idx:03d}" / "mask_warp"
        warped_mask = warp_label_with_transformix(
            label_img=top_segmen[j],
            transform_param_file=final_transform_files[j],
            out_dir=warp_dir,
        )
        warped_masks.append(warped_mask)
    # --------------------------------------------------------
    # Fuse warped masks and compare to GT mask
    # --------------------------------------------------------
    fused_mask = vote_fusion(warped_masks, vote_threshold=VOTE_THRESHOLD)
    dice = dice_score(fused_mask, gt_mask)
    jacc = jaccard_score(fused_mask, gt_mask)
    hd = hausdorff_distance_mm(fused_mask, gt_mask)
    rvd = relative_volume_difference(fused_mask, gt_mask)
    t5 = time.perf_counter()
    # --------------------------------------------------------
    # Plot styling
    # --------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.titlesize": 15
    })
    # --------------------------------------------------------
    # Plot registration results (only kept atlases)
    # --------------------------------------------------------
    n_kept = len(keep)
    n_plots = n_kept + 1  # +1 for the test image
    ncols = min(n_plots, 3)
    nrows = math.ceil(n_plots / ncols)
    fig1, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), dpi=120)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    fig1.suptitle(
        f"Atlases kept for fusion (metric >= {METRIC_THRESHOLD}): {n_kept}",
        fontsize=15,
        y=0.96
    )
    ax = axes[0, 0]
    ax.imshow(fixed_img[:, :, VISUALISATION_SLICE], cmap="gray")
    ax.set_title("Test image", fontsize=13, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for plot_i, j in enumerate(keep):
        pos = plot_i + 1
        row = pos // ncols
        col = pos % ncols
        ax = axes[row, col]
        ax.imshow(final_images[j][:, :, VISUALISATION_SLICE], cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if bs_metrics[j] is None:
            bs_str = "fail"
        else:
            bs_str = f"{bs_metrics[j]:.3f}"
        title_txt = (
            f"Atlas {top_indices[j]:03d}\n"
            f"Affine: {top_metrics[j]:.3f} | BS: {bs_str}\n"
            f"Used: {used_stage[j]} | Final: {final_metrics[j]:.3f}"
        )
        ax.set_title(title_txt, fontsize=11, pad=8)
    # hide unused subplots
    for pos in range(n_plots, nrows * ncols):
        axes[pos // ncols, pos % ncols].axis("off")
    fig1.tight_layout(rect=[0, 0, 1, 0.92])
    # --------------------------------------------------------
    # Plot segmentation result
    # --------------------------------------------------------
    fixed_np = sitk.GetArrayFromImage(fixed_img)
    fused_np = sitk.GetArrayFromImage(binarize(fused_mask))
    gt_np = sitk.GetArrayFromImage(binarize(gt_mask))
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.8), dpi=120)
    fig2.suptitle(
        f"Segmentation result  (threshold={METRIC_THRESHOLD}, kept={n_kept})\n"
        f"Dice: {dice:.3f} | Jaccard: {jacc:.3f} | "
        f"Hausdorff: {hd:.3f} mm | RVD: {rvd:.3f}",
        fontsize=14,
        y=0.97
    )
    axes2[0].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
    axes2[0].set_title("Test image", fontsize=13, pad=8)
    axes2[0].set_xticks([]); axes2[0].set_yticks([])
    axes2[1].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
    axes2[1].imshow(
        np.ma.masked_where(fused_np[VISUALISATION_SLICE] == 0, fused_np[VISUALISATION_SLICE]),
        cmap="Reds",
        alpha=0.4,
    )
    axes2[1].set_title("Fused prediction", fontsize=13, pad=8)
    axes2[1].set_xticks([]); axes2[1].set_yticks([])
    axes2[2].imshow(fixed_np[VISUALISATION_SLICE], cmap="gray")
    axes2[2].imshow(
        np.ma.masked_where(gt_np[VISUALISATION_SLICE] == 0, gt_np[VISUALISATION_SLICE]),
        cmap="Reds",
        alpha=0.4,
    )
    axes2[2].set_title("Ground truth mask", fontsize=13, pad=8)
    axes2[2].set_xticks([]); axes2[2].set_yticks([])
    fig2.subplots_adjust(
        left=0.03,
        right=0.99,
        bottom=0.06,
        top=0.80,
        wspace=0.03
    )
t6 = time.perf_counter()
# ------------------------------------------------------------
# Print timings and summary
# ------------------------------------------------------------
print("___________________________________________")
print(f"Loading the images      : {(t1 - t0)//60:.0f}m {(t1 - t0)%60:.2f}s")
print(f"Affine registrations    : {(t2 - t1)//60:.0f}m {(t2 - t1)%60:.2f}s")
print(f"Preselection            : {(t3 - t2)//60:.0f}m {(t3 - t2)%60:.2f}s")
print(f"BSpline refinements     : {(t4 - t3)//60:.0f}m {(t4 - t3)%60:.2f}s")
print(f"Mask warping + fusion   : {(t5 - t4)//60:.0f}m {(t5 - t4)%60:.2f}s")
print(f"Plotting                : {(t6 - t5)//60:.0f}m {(t6 - t5)%60:.2f}s")
print("___________________________________________")
print(f"Total time              : {(t6 - t0)//60:.0f}m {(t6 - t0)%60:.2f}s")
print(f"\nMetric threshold        : {METRIC_THRESHOLD}")
print(f"Atlases kept for fusion : {len(keep)} / {PRESELECTION_SIZE}")
print("\nAll 5 atlases:")
for i in range(len(top_indices)):
    if bs_metrics[i] is None:
        bs_str = "fail"
    else:
        bs_str = f"{bs_metrics[i]:.6f}"
    kept_str = "KEPT" if i in keep else "DROPPED"
    print(
        f"  Atlas {top_indices[i]:03d} | "
        f"Affine = {top_metrics[i]:.6f} | "
        f"BS = {bs_str} | "
        f"Final = {final_metrics[i]:.6f} | "
        f"Stage = {used_stage[i]} | "
        f"{kept_str}"
    )
print("\nSegmentation metrics:")
print(f"  Dice                  : {dice:.6f}")
print(f"  Jaccard               : {jacc:.6f}")
print(f"  Hausdorff distance mm : {hd:.6f}")
print(f"  Relative volume diff  : {rvd:.6f}")
plt.show()
