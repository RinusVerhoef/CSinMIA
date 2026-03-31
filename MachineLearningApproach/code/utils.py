import random

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF


class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size, valid=False):
        self.valid = valid
        self.mr_image_list = []
        self.mask_list = []
        # load images
        self.no_slices = 24
        for path in paths:
            im = sitk.GetArrayFromImage(sitk.ReadImage(path / "t2.nii.gz")).astype(
                np.int32
            )
            Ns = im.shape[0]
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "t2.nii.gz")).astype(
                    np.int32
                )[Ns // 2 - self.no_slices // 2 : Ns // 2 + self.no_slices // 2, ...]
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(
                    sitk.ReadImage(path / "t2_anatomy_reader1.nii.gz")
                ).astype(np.int32)[
                    Ns // 2 - self.no_slices // 2 : Ns // 2 + self.no_slices // 2, ...
                ]
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)

        if valid:
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    # transforms.CenterCrop(256),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            # transforms to resize images
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    # transforms.CenterCrop(256),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            )

        self.train_data_mean = 287
        self.train_data_std = 212
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.

        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        valid : bool, optional
            whether the data is for validation
        """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)

        x = self.norm_transform(
            self.img_transform(self.mr_image_list[patient][the_slice, ...]).float()
        )
        random.seed(seed)
        torch.manual_seed(seed)
        y = self.img_transform(
            (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
        )
        
        # Augmentation for training data
        if not self.valid:
            # Original image shapes
            h, w = x.shape[-2:], x.shape[-1:]
            # Horizontal mirroring
            if random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

            # Rotation
            angle = random.uniform(-10, 10)
            x = TF.rotate(x, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            y = TF.rotate(y, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # Random cropping
            h, w = x.shape[-2:]
            crop_pct = random.uniform(0.05, 0.10)  # Only small cropping to avoid losing prostate
            ch, cw = int(h * (1 - crop_pct)), int(w * (1 - crop_pct))
            # Pick a random top-left corner for cropping
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            x = TF.crop(x, top, left, ch, cw)
            y = TF.crop(y, top, left, ch, cw)
            # Resize back to original size
            x = TF.resize(x, (h, w))
            y = TF.resize(y, (h, w), interpolation=transforms.InterpolationMode.NEAREST)
        return x, y

class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1

        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            outputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return BCE + dice_loss
