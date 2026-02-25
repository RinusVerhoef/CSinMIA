import os
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Folder that contains patient folders (001, 002, ..., 045, etc.)
BASE_DIR = r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train"
PATIENT_ID = "045"

# Full path to the MRI file
image_path = os.path.join(BASE_DIR, PATIENT_ID, "t2.nii.gz")

# Read image (3D MRI volume)
img = sitk.ReadImage(image_path)

# Convert to NumPy array (shape usually: [slices, height, width])
arr = sitk.GetArrayFromImage(img)

# Convert to PyTorch tensor (optional, only if you need torch later)
x = torch.from_numpy(arr).float()

# Pick the middle slice
middle_slice = x.shape[0] // 2

# Show it
plt.imshow(x[middle_slice].numpy(), cmap="gray")
plt.title(f"Patient {PATIENT_ID} - Slice {middle_slice}")
plt.axis("off")
plt.show()