# registration_efthimis.py

from pathlib import Path
import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from prostateLoader import ProstateLoader


def ncc(a, b, eps=1e-8):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + eps)
    b = (b - b.mean()) / (b.std() + eps)
    return float((a * b).mean())


def dice(a, b, eps=1e-8):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return float((2.0 * inter) / (a.sum() + b.sum() + eps))


def warp_mask_nearest(moving_mask: sitk.Image, transform_parameter_map):
    tmap = transform_parameter_map
    for m in tmap:
        m["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        m["FinalBSplineInterpolationOrder"] = ["0"]
        m["ResultImagePixelType"] = ["unsigned char"]

    tx = sitk.TransformixImageFilter()
    tx.SetTransformParameterMap(tmap)
    tx.SetMovingImage(moving_mask)
    tx.Execute()

    out = tx.GetResultImage()
    out = sitk.Cast(out > 0, sitk.sitkUInt8)
    return out


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "prostate158_train" / "train"
assert DATA_ROOT.exists(), f"Missing dataset folder: {DATA_ROOT}"

LOG_ROOT = Path(r"C:\temp\elastix_logs")
LOG_ROOT.mkdir(parents=True, exist_ok=True)
print("Writing elastix logs to:", LOG_ROOT)


loader = ProstateLoader(root=str(DATA_ROOT))
images, masks = loader.LoadData()

atlas_size = 8
atlas_images = images[:atlas_size]
atlas_masks = masks[:atlas_size]
test_images = images[atlas_size:]
test_masks = masks[atlas_size:]

fixed_img = test_images[0]
fixed_mask = test_masks[0]

fixed_np = sitk.GetArrayFromImage(fixed_img)        # (Z,Y,X)
fixed_mask_np = sitk.GetArrayFromImage(fixed_mask)  # (Z,Y,X)
slice_idx = fixed_np.shape[0] // 2


warped_masks = []
scores = []

for i, (moving_img, moving_mask) in enumerate(zip(atlas_images, atlas_masks)):
    run_dir = LOG_ROOT / f"atlas_{i:02d}_to_fixed"
    run_dir.mkdir(parents=True, exist_ok=True)

    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed_img)
    elx.SetMovingImage(moving_img)

    elx.SetOutputDirectory(str(run_dir))
    elx.LogToConsoleOn()
    elx.LogToFileOn()

    pm_translation = sitk.GetDefaultParameterMap("translation")
    pm_affine = sitk.GetDefaultParameterMap("affine")
    pm_affine["MaximumNumberOfIterations"] = ["256"]

    elx.SetParameterMap(pm_translation)
    elx.AddParameterMap(pm_affine)

    try:
        elx.Execute()
    except RuntimeError:
        print("\nElastix failed. Open this folder and read elastix.log:")
        print(run_dir)
        raise

    reg_img = elx.GetResultImage()
    tmap = elx.GetTransformParameterMap()

    reg_np = sitk.GetArrayFromImage(reg_img)
    score_ncc = ncc(fixed_np, reg_np)

    warped = warp_mask_nearest(moving_mask, tmap)
    warped_np = sitk.GetArrayFromImage(warped)
    score_dice = dice(warped_np, fixed_mask_np)

    warped_masks.append(warped_np)
    scores.append((i, score_ncc, score_dice))

    print(f"Atlas {i:02d}: NCC={score_ncc:.4f}, Dice={score_dice:.4f} | logs: {run_dir}")


K = min(5, len(scores))
top = sorted(scores, key=lambda x: x[1], reverse=True)[:K]
top_idx = [t[0] for t in top]
print("Top-K atlases by NCC:", top_idx)

stack = np.stack([warped_masks[i] for i in top_idx], axis=0)
fused = (stack.sum(axis=0) >= (K / 2)).astype(np.uint8)

fused_dice = dice(fused, fixed_mask_np)
print(f"Fused Dice (top-{K} by NCC): {fused_dice:.4f}")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(fixed_np[slice_idx], cmap="gray")
plt.imshow(fixed_mask_np[slice_idx], cmap="jet", alpha=0.3)
plt.title("Fixed + GT")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(fixed_np[slice_idx], cmap="gray")
plt.imshow(fused[slice_idx], cmap="jet", alpha=0.3)
plt.title(f"Fixed + Fused (Dice {fused_dice:.3f})")
plt.axis("off")

plt.tight_layout()
plt.show(block=True)
input("Press Enter to exit...")