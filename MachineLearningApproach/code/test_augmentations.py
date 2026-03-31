import torch
import matplotlib.pyplot as plt
from utils import ProstateMRDataset
from pathlib import Path
import random

# Load datasets
paths = sorted([p for p in Path("prostate158_train/train").glob("*")])

ds_orig = ProstateMRDataset(paths, img_size=[64, 64], valid=True)   # no augmentation
ds_aug  = ProstateMRDataset(paths, img_size=[64, 64], valid=False)  # augmentation

def show_comparison(orig_img, orig_mask, aug_img, aug_mask):
    orig = orig_img.squeeze().numpy()
    orig_m = orig_mask.squeeze().numpy()
    aug = aug_img.squeeze().numpy()
    aug_m = aug_mask.squeeze().numpy()

    plt.figure(figsize=(10,6))

    # Original
    plt.subplot(2,2,1)
    plt.title("Original Image")
    plt.imshow(orig, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.title("Original Overlay")
    plt.imshow(orig, cmap="gray")
    plt.imshow(orig_m, cmap="Reds", alpha=0.4)
    plt.axis("off")

    # Augmented
    plt.subplot(2,2,3)
    plt.title("Augmented Image")
    plt.imshow(aug, cmap="gray")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.title("Augmented Overlay")
    plt.imshow(aug, cmap="gray")
    plt.imshow(aug_m, cmap="Reds", alpha=0.4)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Show 10 random comparisons
for _ in range(10):
    idx = random.randint(0, len(ds_orig)-1)
    orig_x, orig_y = ds_orig[idx]
    aug_x,  aug_y  = ds_aug[idx]
    show_comparison(orig_x, orig_y, aug_x, aug_y)