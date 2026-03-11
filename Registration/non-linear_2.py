from pathlib import Path
import shutil
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


DEBUG_BSPLINE_DIR = Path("bspline_debug")
DEBUG_BSPLINE_DIR.mkdir(exist_ok=True)


def run_bspline_from_affine(fixed_img, moving_img, affine_tmap, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)

    affine_tp_file = work_dir / "affine_tp.txt"
    sitk.WriteParameterFile(last_parameter_map(affine_tmap), str(affine_tp_file))

    pm_bspline = sitk.ReadParameterFile("ParameterFiles/BSpline/bspline.txt")
    pm_bspline["InitialTransformParameterFileName"] = [str(affine_tp_file).replace("\\", "/")]
    pm_bspline["HowToCombineTransforms"] = ["Compose"]

    print("\n================ BSPLINE DEBUG ================")
    print("Work dir:", work_dir)
    print("Affine TP file:", affine_tp_file)
    print("InitialTransformParameterFileName:", pm_bspline["InitialTransformParameterFileName"])
    print("HowToCombineTransforms:", pm_bspline["HowToCombineTransforms"])
    print("================================================\n")

    elx_bspline = sitk.ElastixImageFilter()
    elx_bspline.SetFixedImage(fixed_img)
    elx_bspline.SetMovingImage(moving_img)
    elx_bspline.SetParameterMap(pm_bspline)
    elx_bspline.SetOutputDirectory(str(work_dir))
    elx_bspline.LogToConsoleOn()
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
DEBUG_TOP_K = 0
TEST_PATIENT_INDEX = 3

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
# Select fixed image and GT mask
# ------------------------------------------------------------
fixed_img = test_images[TEST_PATIENT_INDEX]
gt_mask = test_segmen[TEST_PATIENT_INDEX]

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

print("Running affine registration for all atlases...")

for idx, moving_img in enumerate(atlas_images):
    print(f"Affine atlas {idx:03d}")
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

print("\nTop 5 affine atlases:")
for i in range(len(top_indices)):
    print(f"Rank {i+1}: atlas {top_indices[i]:03d} | affine metric = {top_metrics[i]:.6f}")

# ------------------------------------------------------------
# Debug only one BSpline case
# ------------------------------------------------------------
atlas_idx = top_indices[DEBUG_TOP_K]
moving_img = top_images[DEBUG_TOP_K]
moving_seg = top_segmen[DEBUG_TOP_K]
affine_tmap = top_tmaps[DEBUG_TOP_K]
affine_img = top_reg[DEBUG_TOP_K]

work_dir = DEBUG_BSPLINE_DIR / f"atlas_{atlas_idx:03d}_{int(time.time())}"
work_dir.mkdir(parents=True, exist_ok=True)

print("\nTesting BSpline only for one atlas...")
print(f"Selected rank: {DEBUG_TOP_K + 1}")
print(f"Atlas index: {atlas_idx:03d}")
print(f"Affine metric: {top_metrics[DEBUG_TOP_K]:.6f}")
print(f"Debug folder: {work_dir}")

affine_tp_file = write_transform_parameter_file_from_tmap(
    affine_tmap,
    work_dir / "affine_only_tp.txt"
)

bspline_ok = False
bs_metric = None
bs_tp_file = None
bspline_img = None

try:
    bspline_img, bs_metric, bs_tp_file = run_bspline_from_affine(
        fixed_img=fixed_img,
        moving_img=moving_img,
        affine_tmap=affine_tmap,
        work_dir=work_dir,
    )

    bspline_ok = True
    print("\nBSpline worked")
    print("BS metric:", bs_metric)
    print("Transform file:", bs_tp_file)

except Exception as e:
    print("\nBSpline failed")
    print("Exception:")
    print(repr(e))

    log_file = work_dir / "elastix.log"
    print("Log exists:", log_file.exists())

    if log_file.exists():
        txt = log_file.read_text(encoding="utf-8", errors="ignore")
        print("\n================ END OF elastix.log ================\n")
        print(txt[-4000:])
        print("\n====================================================\n")

t4 = time.perf_counter()

# ------------------------------------------------------------
# Optional visual comparison
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.titlesize": 15
})

fig, axes = plt.subplots(1, 3 if bspline_ok else 2, figsize=(15, 5), dpi=120)

axes[0].imshow(fixed_img[:, :, VISUALISATION_SLICE], cmap="gray")
axes[0].set_title("Fixed test image")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(affine_img[:, :, VISUALISATION_SLICE], cmap="gray")
axes[1].set_title(
    f"Affine result\nAtlas {atlas_idx:03d} | metric: {top_metrics[DEBUG_TOP_K]:.3f}"
)
axes[1].set_xticks([])
axes[1].set_yticks([])

if bspline_ok:
    axes[2].imshow(bspline_img[:, :, VISUALISATION_SLICE], cmap="gray")
    axes[2].set_title(
        f"BSpline result\nAtlas {atlas_idx:03d} | metric: {bs_metric:.3f}"
    )
    axes[2].set_xticks([])
    axes[2].set_yticks([])

fig.suptitle("Affine vs BSpline debug", y=0.95)
fig.subplots_adjust(left=0.03, right=0.99, bottom=0.08, top=0.83, wspace=0.05)

t5 = time.perf_counter()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("___________________________________________")
print(f"Loading the images      : {(t1 - t0)//60:.0f}m {(t1 - t0)%60:.2f}s")
print(f"Affine registrations    : {(t2 - t1)//60:.0f}m {(t2 - t1)%60:.2f}s")
print(f"Preselection            : {(t3 - t2)//60:.0f}m {(t3 - t2)%60:.2f}s")
print(f"BSpline debug           : {(t4 - t3)//60:.0f}m {(t4 - t3)%60:.2f}s")
print(f"Plotting                : {(t5 - t4)//60:.0f}m {(t5 - t4)%60:.2f}s")
print("___________________________________________")
print(f"Total time              : {(t5 - t0)//60:.0f}m {(t5 - t0)%60:.2f}s")

print("\nSelected top 5 atlases:")
for i in range(len(top_indices)):
    print(
        f"Rank {i+1} | "
        f"Atlas {top_indices[i]:03d} | "
        f"Affine metric = {top_metrics[i]:.6f}"
    )

print("\nBSpline debug result:")
print(f"Test patient index      : {TEST_PATIENT_INDEX}")
print(f"Selected top rank       : {DEBUG_TOP_K + 1}")
print(f"Selected atlas          : {atlas_idx:03d}")
print(f"Affine metric           : {top_metrics[DEBUG_TOP_K]:.6f}")
print(f"BSpline success         : {bspline_ok}")

if bspline_ok:
    print(f"BSpline metric          : {bs_metric:.6f}")
    print(f"BSpline transform file  : {bs_tp_file}")
else:
    print(f"Debug folder            : {work_dir}")
    print("Check affine_only_tp.txt and elastix.log inside that folder.")

plt.show()