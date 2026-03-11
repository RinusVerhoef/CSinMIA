from pathlib import Path
from tempfile import TemporaryDirectory
import re
import time

from prostateLoader import ProstateLoader
from utils import final_metric_from_elastix_log

import SimpleITK as sitk
import matplotlib.pyplot as plt


def final_metric_from_log(log_path):
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


def run_bspline_from_affine(fixed_img, moving_img, affine_tmap, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)

    affine_tp_file = work_dir / "affine_tp.txt"
    sitk.WriteParameterFile(last_parameter_map(affine_tmap), str(affine_tp_file))

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

    return registered_img, bs_metric


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
# Select fixed image
# ------------------------------------------------------------
fixed_img = test_images[1]

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
# ------------------------------------------------------------
final_images = []
bs_metrics = []
used_stage = []

with TemporaryDirectory(prefix="bspline_top5_") as tmp:
    tmp_root = Path(tmp)

    for i, moving_img in enumerate(top_images):
        atlas_idx = top_indices[i]
        work_dir = tmp_root / f"atlas_{atlas_idx:03d}"

        try:
            bspline_img, bs_metric = run_bspline_from_affine(
                fixed_img=fixed_img,
                moving_img=moving_img,
                affine_tmap=top_tmaps[i],
                work_dir=work_dir,
            )

            if bs_metric > top_metrics[i]:
                final_images.append(bspline_img)
                bs_metrics.append(bs_metric)
                used_stage.append("BS")
            else:
                final_images.append(top_reg[i])
                bs_metrics.append(bs_metric)
                used_stage.append("AFF")

        except Exception as e:
            print(f"BSpline failed for atlas {atlas_idx:03d}, keeping affine: {e}")
            final_images.append(top_reg[i])
            bs_metrics.append(None)
            used_stage.append("AFF")

t4 = time.perf_counter()

# ------------------------------------------------------------
# Clean plotting
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "figure.titlesize": 18
})

fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=120)
fig.suptitle("Top 5 atlases after affine preselection and BSpline refinement", y=0.98)

# Fixed image
ax = axes[0, 0]
ax.imshow(fixed_img[:, :, VISUALISATION_SLICE], cmap="gray")
ax.set_title("Test image", pad=10)
ax.set_xticks([])
ax.set_yticks([])

# Final 5 images
for i, img in enumerate(final_images):
    row = (i + 1) // 3
    col = (i + 1) % 3
    ax = axes[row, col]

    ax.imshow(img[:, :, VISUALISATION_SLICE], cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    if bs_metrics[i] is None:
        title_txt = (
            f"Atlas {top_indices[i]:03d}\n"
            f"Affine: {top_metrics[i]:.3f}   BS: fail   Used: {used_stage[i]}"
        )
    else:
        title_txt = (
            f"Atlas {top_indices[i]:03d}\n"
            f"Affine: {top_metrics[i]:.3f}   BS: {bs_metrics[i]:.3f}   Used: {used_stage[i]}"
        )

    ax.set_title(title_txt, pad=10)

# In case fewer than 5 images ever appear
for k in range(len(final_images) + 1, 6):
    row = k // 3
    col = k % 3
    axes[row, col].axis("off")

fig.tight_layout(rect=[0, 0.02, 1, 0.95])

t5 = time.perf_counter()

# ------------------------------------------------------------
# Print timings and summary
# ------------------------------------------------------------
print("___________________________________________")
print(f"Loading the images      : {(t1 - t0)//60:.0f}m {(t1 - t0)%60:.2f}s")
print(f"Affine registrations    : {(t2 - t1)//60:.0f}m {(t2 - t1)%60:.2f}s")
print(f"Preselection            : {(t3 - t2)//60:.0f}m {(t3 - t2)%60:.2f}s")
print(f"BSpline refinements     : {(t4 - t3)//60:.0f}m {(t4 - t3)%60:.2f}s")
print(f"Plotting                : {(t5 - t4)//60:.0f}m {(t5 - t4)%60:.2f}s")
print("___________________________________________")
print(f"Total time              : {(t5 - t0)//60:.0f}m {(t5 - t0)%60:.2f}s")

print("\nSelected atlases:")
for i in range(len(top_indices)):
    if bs_metrics[i] is None:
        print(
            f"Atlas {top_indices[i]:03d} | "
            f"Affine metric = {top_metrics[i]:.6f} | "
            f"BS metric = fail | "
            f"Used = {used_stage[i]}"
        )
    else:
        print(
            f"Atlas {top_indices[i]:03d} | "
            f"Affine metric = {top_metrics[i]:.6f} | "
            f"BS metric = {bs_metrics[i]:.6f} | "
            f"Used = {used_stage[i]}"
        )

plt.show()