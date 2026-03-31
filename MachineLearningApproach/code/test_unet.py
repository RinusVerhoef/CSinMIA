import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import DataLoader
import u_net
import utils
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = [64, 64]
BATCH_SIZE = 1

TEST_DIR = Path.cwd() / "prostate158_train" / "test"

MODEL_PATHS = {
    "baseline": "MachineLearningApproach\\unet_weights\\baseline_unet.pth",
    "random_aug": "MachineLearningApproach\\unet_weights\\random_unet.pth",
    "targeted_aug": "MachineLearningApproach\\unet_weights\\targeted_unet.pth",
}

# Dataset
patients_test = [
    p for p in TEST_DIR.glob("*")
    if not any(part.startswith(".") for part in p.parts)
]

test_dataset = utils.ProstateMRDataset(patients_test, IMAGE_SIZE, valid=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Evaluation metrics
def dice(pred, target, smooth=1):
    pred = (pred > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def hd95(pred_mask, gt_mask):
    # Handle empty masks — avoid SimpleITK crash
    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 0.0  # perfect match (both empty)
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 100.0  # extremely bad match — large penalty

    pred_img = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    gt_img = sitk.GetImageFromArray(gt_mask.astype(np.uint8))

    # Compute contour images
    pred_surface = sitk.LabelContour(pred_img)
    gt_surface = sitk.LabelContour(gt_img)

    # Compute distance maps
    pred_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_surface, squaredDistance=False))
    gt_dm = sitk.Abs(sitk.SignedMaurerDistanceMap(gt_surface, squaredDistance=False))

    pred_dm_np = sitk.GetArrayViewFromImage(pred_dm)
    gt_dm_np = sitk.GetArrayViewFromImage(gt_dm)

    pred_pts = np.where(pred_surface != 0)
    gt_pts = np.where(gt_surface != 0)

    if len(pred_pts[0]) == 0 or len(gt_pts[0]) == 0:
        return 100.0  # no surface → treat as large error

    d_pred_to_gt = gt_dm_np[pred_pts]
    d_gt_to_pred = pred_dm_np[gt_pts]

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return np.percentile(all_dists, 95)

# Evaluation loop
def evaluate_model(model_path):
    print(f"\n=== Evaluating model: {model_path} ===")

    # load model
    model = u_net.UNet(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dice_scores = []
    hd95_scores = []

    # folder for saving visual examples
    example_dir = Path("MachineLearningApproach\\code\\figures") / model_name
    example_dir.mkdir(parents=True, exist_ok=True)

    # choose one random index to save
    example_index = random.randint(0, len(test_dataset) - 1)

    with torch.no_grad():
        for idx, (img, mask) in enumerate(test_loader):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            out = torch.sigmoid(model(img))
            pred = (out > 0.5).float()

            # compute metrics
            d = dice(pred, mask)
            dice_scores.append(d.item())

            pred_np = pred.cpu().numpy()[0, 0]
            mask_np = mask.cpu().numpy()[0, 0]
            
            if pred_np.sum() == 0 or mask_np.sum() == 0:
                hd95_scores.append(100.0)
                continue

            h95 = hd95(pred_np, mask_np)
            hd95_scores.append(h95)

           # SAVE EXAMPLE
            if idx == example_index:
                img_np = img.cpu().numpy()[0, 0]
                pred_vis = pred_np
                gt_vis = mask_np

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img_np, cmap="gray")
                axs[0].set_title("MRI Slice")
                axs[1].imshow(gt_vis, cmap="gray")
                axs[1].set_title("Ground Truth Mask")
                axs[2].imshow(pred_vis, cmap="gray")
                axs[2].set_title("Predicted Mask")

                for a in axs:
                    a.axis("off")

                savepath = example_dir / "example_segmentation.png"
                plt.savefig(savepath, dpi=150, bbox_inches="tight")
                plt.close()

    return dice_scores, hd95_scores

# Run evaluation for all models
for model_name, model_path in MODEL_PATHS.items():
    dice_list, hd95_list = evaluate_model(model_path)

    print(f"\n### Results for {model_name} ###")
    print(f"Mean Dice: {np.mean(dice_list):.4f}")
    print(f"Median Dice: {np.median(dice_list):.4f}")
    print(f"Mean HD95: {np.mean(hd95_list):.2f} px")
    print(f"Median HD95: {np.median(hd95_list):.2f} px")
    print("--------------------------------------")