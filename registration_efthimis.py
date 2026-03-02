from pathlib import Path
import numpy as np
import SimpleITK as sitk
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


def sanitize(pm):
    # In your TransformParameters.1.txt the line is:
    # (InitialTransformParameterFileName "./TransformParameters.0.txt")
    # Make it self-contained:
    if "InitialTransformParameterFileName" in pm:
        pm["InitialTransformParameterFileName"] = ["NoInitialTransform"]
    return pm


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "prostate158_train" / "train"

PM0 = PROJECT_ROOT / "TransformParameters.0.txt"
PM1 = PROJECT_ROOT / "TransformParameters.1.txt"

assert DATA_ROOT.exists(), f"Missing dataset folder: {DATA_ROOT}"
assert PM0.exists(), f"Missing parameter file: {PM0}"
assert PM1.exists(), f"Missing parameter file: {PM1}"


# -----------------------------
# Load data
# -----------------------------
loader = ProstateLoader(root=str(DATA_ROOT))
images, masks = loader.LoadData()

atlas_size = 8
atlas_images = images[:atlas_size]
atlas_masks = masks[:atlas_size]
test_images = images[atlas_size:]
test_masks = masks[atlas_size:]

fixed_img = test_images[0]
fixed_mask = test_masks[0]

fixed_np = sitk.GetArrayFromImage(fixed_img)
fixed_mask_np = sitk.GetArrayFromImage(fixed_mask)
slice_idx = fixed_np.shape[0] // 2


# -----------------------------
# Register each atlas -> fixed, warp masks
# -----------------------------
warped_masks = []
scores = []  # (i, ncc, dice)

for i, (moving_img, moving_mask) in enumerate(zip(atlas_images, atlas_masks)):
    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed_img)
    elx.SetMovingImage(moving_img)

    pm0 = sanitize(sitk.ReadParameterFile(str(PM0)))
    pm1 = sanitize(sitk.ReadParameterFile(str(PM1)))

    elx.SetParameterMap(pm0)
    elx.AddParameterMap(pm1)

    elx.LogToConsoleOff()
    elx.Execute()

    reg_img = elx.GetResultImage()
    tmap = elx.GetTransformParameterMap()

    reg_np = sitk.GetArrayFromImage(reg_img)
    s_ncc = ncc(fixed_np, reg_np)

    warped = warp_mask_nearest(moving_mask, tmap)
    warped_np = sitk.GetArrayFromImage(warped)
    s_dice = dice(warped_np, fixed_mask_np)

    warped_masks.append(warped_np)
    scores.append((i, s_ncc, s_dice))

    print(f"Atlas {i:02d}: NCC={s_ncc:.4f}, Dice={s_dice:.4f}")


# -----------------------------
# Fuse top-K by NCC
# -----------------------------
K = min(5, len(scores))
top = sorted(scores, key=lambda x: x[1], reverse=True)[:K]
top_idx = [t[0] for t in top]
print("Top-K atlases by NCC:", top_idx)

stack = np.stack([warped_masks[i] for i in top_idx], axis=0)
fused = (stack.sum(axis=0) >= (K / 2)).astype(np.uint8)

fused_dice = dice(fused, fixed_mask_np)
print(f"Fused Dice (top-{K} by NCC): {fused_dice:.4f}")


# -----------------------------
# Plot overlay
# -----------------------------
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
plt.show()