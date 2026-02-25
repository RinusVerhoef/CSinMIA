import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Settings
BASE_DIR = r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train"
OUTPUT_DIR = r"C:\Users\30697\OneDrive\2.Netherlands\capita\slices"
PATIENT_ID = "027"

# Paths
image_path = os.path.join(BASE_DIR, PATIENT_ID, "t2.nii.gz")
patient_output_dir = os.path.join(OUTPUT_DIR, f"patient_{PATIENT_ID}")

# Checks and folder creation
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"MRI file not found:\n{image_path}")

os.makedirs(patient_output_dir, exist_ok=True)

# Load 3D MRI volume (shape usually: [slices, height, width])
img = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

print("Loaded:", image_path)
print("Volume shape:", img.shape)

# Export all slices as PNG images
for i in range(img.shape[0]):
    plt.figure(figsize=(6, 6))
    plt.imshow(img[i], cmap="gray")
    plt.title(f"Patient {PATIENT_ID} - Slice {i}")
    plt.axis("off")

    save_path = os.path.join(patient_output_dir, f"slice_{i:03d}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

print(f"All slices saved in: {patient_output_dir}")