import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Import nifti files of prostate MRI and segmentation
im = nib.load(f"prostate158_train/train/020/t2.nii.gz").get_fdata()
seg = nib.load("prostate158_train/train/020/t2_anatomy_reader1.nii.gz").get_fdata()

# Isolate the gland mask (label 1 = gland)
gland_mask = (seg == 1).astype(np.uint8)

# Gland only image
gland_only_image = im * gland_mask

# pick a slice index
slice_idx = im.shape[2] // 2   # middle slice

# Plot image with segmentation overlay
plt.imshow(im[:, :, slice_idx], cmap='gray')
plt.imshow(gland_mask[:, :, slice_idx], cmap='jet', alpha=0.3)
plt.imshow(seg[:, :, slice_idx], cmap='jet', alpha=0.3)
plt.axis('off')
plt.show()


# Plot gland only image
# plt.imshow(gland_only_image[:, :, slice_idx], cmap='gray')
# plt.axis('off')
# plt.show()
