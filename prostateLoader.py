import nibabel as nib
import numpy as np
from pathlib import Path


class ProstateLoader:
    def __init__(self, root="prostate158_train/train"):
        self.root = Path(root)
        pass

    def LoadData(self):
        images = []
        segmentations = []

        # Loop over each patient folder
        for patient_dir in sorted(self.root.iterdir()):
            if patient_dir.is_dir():

                img_path = patient_dir / "t2.nii.gz"
                seg_path = patient_dir / "t2_anatomy_reader1.nii.gz"

                if img_path.exists() and seg_path.exists():
                    img = nib.load(str(img_path)).get_fdata()
                    seg = nib.load(str(seg_path)).get_fdata()

                    mask = (seg == 1).astype(np.uint8)

                    images.append(img)
                    segmentations.append(mask)

        print("Loaded volumes:", len(images))
        print("Example shape:", images[0].shape)

        return images, segmentations
