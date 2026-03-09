import nibabel as nib
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

root = Path("prostate158_train/train")

images = []
segmentations = []
gland_masks = []

# Loop over each patient folder
for patient_dir in sorted(root.iterdir()):
    if patient_dir.is_dir():

        img_path = patient_dir / "t2.nii.gz"
        seg_path = patient_dir / "t2_anatomy_reader1.nii.gz"

        if img_path.exists() and seg_path.exists():
            img = nib.load(str(img_path)).get_fdata()
            seg = nib.load(str(seg_path)).get_fdata()

            gland_mask = (seg == 1).astype(np.uint8)

            images.append(img)
            segmentations.append(seg)
            gland_masks.append(gland_mask)

print("Loaded volumes:", len(images))
print("Example shape:", images[0].shape)

# Choose the image that you want to visualize
idx = 50
im = images[idx]
gland_mask = gland_masks[idx]
seg = segmentations[idx]


# Pick a slice you want to visualize
slice_idx = im.shape[2] // 2  # middle slice

# Plot image with segmentation overlay
plt.figure()
plt.imshow(im[:, :, slice_idx].T, cmap="gray")
plt.imshow(gland_mask[:, :, slice_idx].T, cmap="jet", alpha=0.3)
# plt.imshow(seg[:, :, slice_idx].T, cmap="jet", alpha=0.3)
plt.axis("off")
plt.show()
