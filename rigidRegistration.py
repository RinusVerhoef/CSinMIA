from prostateLoader import ProstateLoader

import matplotlib.pyplot as plt

import SimpleITK as sitk
import os, re
import time


def final_metric_from_elastix_log():
    log_path = "elastix.log"
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No elastix.log found in current dir")

    pat = re.compile(r"Final metric value\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    metric = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                metric = float(m.group(1))
    if metric is None:
        raise RuntimeError("Could not find 'Final metric value' in elastix.log")
    return -metric


t0 = time.perf_counter()

# Load data
loader = ProstateLoader()
images, segmen = loader.LoadData()

# Divide into atlas and test images
atlas_size = 24

atlas_images = images[0:atlas_size]
atlas_segmen = segmen[0:atlas_size]

test_images = images[atlas_size:]
test_segmen = segmen[atlas_size:]


# Visualize
img = sitk.GetArrayFromImage(atlas_images[-1])
seg = sitk.GetArrayFromImage(atlas_segmen[-1])

# Pick a slice you want to visualize
slice_idx = img.shape[0] // 2  # middle slice
print(img.shape)

# Plot image with segmentation overlay
plt.figure()
plt.imshow(img[slice_idx, :, :], cmap="gray")
plt.imshow(seg[slice_idx, :, :], cmap="jet", alpha=0.3)
plt.axis("off")

t1 = time.perf_counter()

fixed_img = test_images[1]

fig, axes = plt.subplots(5, 5, figsize=(12, 4))
img = sitk.GetArrayFromImage(fixed_img)
axes[0, 0].imshow(img[slice_idx, :, :], cmap="gray")
axes[0, 0].set_title("Test Image")

# Elastix registration object
elx = sitk.ElastixImageFilter()
elx.SetFixedImage(fixed_img)

# Load parameter maps from file
pm_rigid = sitk.ReadParameterFile("ParameterFiles/Rigid1/affine.txt")

# Set multi-stage registration
elx.SetParameterMap(pm_rigid)
elx.LogToConsoleOn()
elx.LogToFileOn()

metrics = []

for idx, moving_img in enumerate(atlas_images):

    # Run
    elx.SetMovingImage(moving_img)
    elx.Execute()

    registered_img = elx.GetResultImage()
    tmap = elx.GetTransformParameterMap()  # save this to apply to masks or other images

    fixed_np = sitk.GetArrayFromImage(fixed_img)  # (Z,Y,X)
    reg_np = sitk.GetArrayFromImage(registered_img)  # (Z,Y,X)

    metric = final_metric_from_elastix_log()
    metrics.append(metric)

    img = sitk.GetArrayFromImage(registered_img)
    ax = (idx + 1) % 5
    ay = (idx + 1) // 5
    axes[ax, ay].imshow(img[slice_idx, :, :], cmap="gray")
    axes[ax, ay].set_title(f"NCC: {metric:.3f}")

t2 = time.perf_counter()

print("___________________________________________")
print(f"Loading the images      : {t1 - t0:.3f} s")
print(f"Rigid registrations     : {t2 - t1:.3f} s")
print(f"Total time              : {t2 - t0:.3f} s")

for a in axes.flatten():
    a.axis("off")
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.2)
plt.show()
