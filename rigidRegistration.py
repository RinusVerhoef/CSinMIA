from prostateLoader import ProstateLoader

import matplotlib.pyplot as plt

import SimpleITK as sitk
import numpy as np


def ncc(a, b, eps=1e-8):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + eps)
    b = (b - b.mean()) / (b.std() + eps)
    return float((a * b).mean())


# Load data
loader = ProstateLoader()
images, segmen = loader.LoadData()

# Divide into atlas and test images
atlas_size = 8

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
plt.show()

fixed_img = test_images[0]

plt.figure()

plt.subplot(3, 3, 1)
img = sitk.GetArrayFromImage(fixed_img)
plt.imshow(img[slice_idx, :, :], cmap="gray")

for idx, moving_img in enumerate(atlas_images):

    # Elastix registration object
    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed_img)
    elx.SetMovingImage(moving_img)

    # Choose registration model(s)
    pm_translation = sitk.GetDefaultParameterMap("translation")
    pm_rigid = sitk.GetDefaultParameterMap("affine")

    # Optional: tweak iterations (often helpful)
    pm_rigid["MaximumNumberOfIterations"] = ["512"]

    # Apply parameter maps (multi-stage registration)
    elx.SetParameterMap(pm_translation)
    elx.AddParameterMap(pm_rigid)

    # Run
    elx.LogToConsoleOn()
    elx.Execute()

    registered_img = elx.GetResultImage()
    tmap = elx.GetTransformParameterMap()  # save this to apply to masks or other images

    fixed_np = sitk.GetArrayFromImage(fixed_img)  # (Z,Y,X)
    reg_np = sitk.GetArrayFromImage(registered_img)  # (Z,Y,X)

    metric = ncc(fixed_np, reg_np)

    plt.subplot(3, 3, idx + 2)
    img = sitk.GetArrayFromImage(registered_img)
    plt.imshow(img[slice_idx, :, :], cmap="gray")
    plt.title(f"NCC: {metric}")

plt.show()
