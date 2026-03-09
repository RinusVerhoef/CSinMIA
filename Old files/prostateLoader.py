import SimpleITK as sitk
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
                    img = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
                    seg = sitk.ReadImage(str(seg_path), sitk.sitkFloat32)

                    mask = seg == 1

                    images.append(img)
                    segmentations.append(mask)

        print("Loaded volumes:", len(images))
        print("Example shape:", images[0].GetSize())

        return images, segmentations
