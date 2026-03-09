from pathlib import Path
from datetime import datetime

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from prostateLoader import ProstateLoader


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_unique_output_folder(base_dir: Path, prefix: str = "gt_mask_overlays") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{prefix}_{timestamp}"

    counter = 1
    while out_dir.exists():
        out_dir = base_dir / f"{prefix}_{timestamp}_{counter:02d}"
        counter += 1

    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def save_overlay_image(
    image: sitk.Image,
    mask: sitk.Image,
    save_path: Path,
    patient_idx: int,
) -> None:
    img_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(sitk.Cast(mask > 0, sitk.sitkUInt8))

    slice_idx = img_np.shape[0] // 2

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np[slice_idx], cmap="gray")
    plt.imshow(
        np.ma.masked_where(mask_np[slice_idx] == 0, mask_np[slice_idx]),
        cmap="Reds",
        alpha=0.45,
    )
    plt.title(f"Patient {patient_idx:03d}, slice {slice_idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def export_all_gt_mask_overlays(root: Path, base_output_dir: Path) -> None:
    loader = ProstateLoader(str(root))
    images, masks = loader.LoadData()

    out_dir = create_unique_output_folder(base_output_dir)

    print(f"Saving overlay images to:\n{out_dir}\n")
    print(f"Total patients: {len(images)}\n")

    for i, (img, msk) in enumerate(zip(images, masks)):
        save_path = out_dir / f"patient_{i:03d}_gt_overlay.png"
        save_overlay_image(
            image=img,
            mask=msk,
            save_path=save_path,
            patient_idx=i,
        )
        print(f"Saved patient {i:03d}")

    info_txt = out_dir / "info.txt"
    info_txt.write_text(
        "Ground-truth mask overlay export\n"
        "One PNG per patient\n"
        "Displayed slice = middle slice of the 3D volume\n"
        "Mask color = red\n",
        encoding="utf-8",
    )

    print("\nDone.")


if __name__ == "__main__":
    ROOT = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train")
    BASE_OUTPUT_DIR = Path(r"C:\temp\prostate_gt_mask_overlays")

    export_all_gt_mask_overlays(ROOT, BASE_OUTPUT_DIR)