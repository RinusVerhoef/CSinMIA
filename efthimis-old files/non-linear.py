import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from prostateLoader import ProstateLoader


# ----------------------------
# Utilities
# ----------------------------
def ensure_clean_dir(p: Path) -> Path:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_elastix_path(p: Path) -> str:
    return str(p).replace("\\", "/")


def latest_tp(out_dir: Path) -> Path:
    tps = sorted(out_dir.glob("TransformParameters.*.txt"))
    if not tps:
        raise FileNotFoundError(f"No TransformParameters.*.txt found in: {out_dir}")
    return tps[-1]


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)


def dice_score(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(sitk.Cast(pred > 0, sitk.sitkUInt8)).astype(np.uint8)
    g = sitk.GetArrayFromImage(sitk.Cast(gt > 0, sitk.sitkUInt8)).astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    return 1.0 if den == 0 else (2.0 * inter / den)


# ----------------------------
# Parameter maps
# ----------------------------
def pm_translation() -> dict:
    pm = sitk.GetDefaultParameterMap("translation")
    pm["MaximumNumberOfIterations"] = ["256"]
    pm["WriteIterationInfo"] = ["false"]
    pm["WriteResultImage"] = ["true"]
    pm["ResultImageFormat"] = ["nii"]
    pm["AutomaticTransformInitialization"] = ["true"]
    pm["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    return pm


def pm_affine(initial_tp: Path) -> dict:
    pm = sitk.GetDefaultParameterMap("affine")
    pm["MaximumNumberOfIterations"] = ["256"]
    pm["WriteIterationInfo"] = ["false"]
    pm["WriteResultImage"] = ["true"]
    pm["ResultImageFormat"] = ["nii"]
    pm["AutomaticTransformInitialization"] = ["true"]
    pm["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    pm["InitialTransformParameterFileName"] = [to_elastix_path(initial_tp)]
    pm["HowToCombineTransforms"] = ["Compose"]
    return pm


def pm_bspline(initial_tp: Path) -> dict:
    pm = sitk.GetDefaultParameterMap("bspline")

    pm["NumberOfResolutions"] = ["4"]
    schedule_1d = ["2.803221", "1.988100", "1.410000", "1.000000"]
    pm["GridSpacingSchedule"] = [v for s in schedule_1d for v in (s, s, s)]  # 12 entries

    pm["FinalGridSpacingInPhysicalUnits"] = ["20.0", "20.0", "20.0"]
    pm["MaximumNumberOfIterations"] = ["256"]

    pm["ImageSampler"] = ["RandomCoordinate"]
    pm["NumberOfSpatialSamples"] = ["4096"]
    pm["NewSamplesEveryIteration"] = ["true"]
    pm["CheckNumberOfSamples"] = ["true"]
    pm["MaximumNumberOfSamplingAttempts"] = ["4"]

    pm["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    pm["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
    pm["Metric0Weight"] = ["1.0"]
    pm["Metric1Weight"] = ["1.0"]

    pm["InitialTransformParameterFileName"] = [to_elastix_path(initial_tp)]
    pm["HowToCombineTransforms"] = ["Compose"]

    pm["WriteIterationInfo"] = ["false"]
    pm["WriteResultImage"] = ["true"]
    pm["ResultImageFormat"] = ["nii"]

    nres = int(pm["NumberOfResolutions"][0])
    if len(pm["GridSpacingSchedule"]) not in (nres, nres * 3):
        raise ValueError(
            f"Invalid schedule: NumberOfResolutions={nres}, "
            f"GridSpacingSchedule length={len(pm['GridSpacingSchedule'])}"
        )
    return pm


# ----------------------------
# Registration + mask warp
# ----------------------------
def register_atlas_to_fixed(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    out_dir: Path,
) -> Path:
    """
    Runs translation -> affine -> bspline.
    Returns path to final TransformParameters file (copy in _param_files).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pm_dir = out_dir / "_param_files"
    pm_dir.mkdir(parents=True, exist_ok=True)

    moving_img = resample_to_reference(moving_img, fixed_img)

    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed_img)
    elx.SetMovingImage(moving_img)
    elx.SetOutputDirectory(str(out_dir))
    elx.LogToConsoleOff()
    elx.LogToFileOn()
    elx.SetLogFileName(str(log_dir / "elastix.log"))

    elx.SetParameterMap(pm_translation())
    elx.Execute()
    tp0 = latest_tp(out_dir)
    tp0_copy = pm_dir / "TP.translation.txt"
    tp0_copy.write_text(tp0.read_text(encoding="utf-8"), encoding="utf-8")

    elx.SetParameterMap(pm_affine(tp0_copy))
    elx.Execute()
    tp1 = latest_tp(out_dir)
    tp1_copy = pm_dir / "TP.affine.txt"
    tp1_copy.write_text(tp1.read_text(encoding="utf-8"), encoding="utf-8")

    pm3 = pm_bspline(tp1_copy)
    sitk.WriteParameterFile(pm3, str(pm_dir / "bspline_used.txt"))
    elx.SetParameterMap(pm3)
    elx.Execute()

    registered = elx.GetResultImage()
    sitk.WriteImage(registered, str(out_dir / "registered.nii.gz"))

    final_tp = latest_tp(out_dir)
    final_tp_copy = pm_dir / "TP.final.txt"
    final_tp_copy.write_text(final_tp.read_text(encoding="utf-8"), encoding="utf-8")
    return final_tp_copy


def warp_label_with_transformix(label_img: sitk.Image, final_tp_file: Path, out_dir: Path) -> sitk.Image:
    out_dir.mkdir(parents=True, exist_ok=True)

    tp_text = final_tp_file.read_text(encoding="utf-8")

    # enforce nearest-neighbor for masks
    if '(ResampleInterpolator "FinalBSplineInterpolator")' in tp_text:
        tp_text = tp_text.replace(
            '(ResampleInterpolator "FinalBSplineInterpolator")',
            '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
        )
    elif '(ResampleInterpolator "FinalLinearInterpolator")' in tp_text:
        tp_text = tp_text.replace(
            '(ResampleInterpolator "FinalLinearInterpolator")',
            '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
        )

    tp_nn = out_dir / "TP.final.NN.txt"
    tp_nn.write_text(tp_text, encoding="utf-8")

    tfx = sitk.TransformixImageFilter()
    tfx.SetMovingImage(label_img)
    tfx.SetTransformParameterMap(sitk.ReadParameterFile(str(tp_nn)))
    tfx.SetOutputDirectory(str(out_dir))
    tfx.LogToConsoleOff()
    tfx.LogToFileOff()
    tfx.Execute()

    warped = tfx.GetResultImage()
    warped = sitk.Cast(warped > 0.5, sitk.sitkUInt8)
    sitk.WriteImage(warped, str(out_dir / "warped_mask.nii.gz"))
    return warped


def majority_vote(masks: list[sitk.Image]) -> sitk.Image:
    arrs = [sitk.GetArrayFromImage(m).astype(np.uint8) for m in masks]
    stack = np.stack(arrs, axis=0)
    votes = stack.sum(axis=0)
    fused = (votes >= (len(masks) / 2.0)).astype(np.uint8)
    fused_img = sitk.GetImageFromArray(fused)
    fused_img.CopyInformation(masks[0])
    return fused_img


# ----------------------------
# Experiment: 8 atlas, rest test
# ----------------------------
def run_full_experiment(root: Path, out_root: Path, atlas_size: int = 8) -> None:
    loader = ProstateLoader(str(root))
    images, masks = loader.LoadData()

    atlas_images = images[:atlas_size]
    atlas_masks = masks[:atlas_size]
    test_images = images[atlas_size:]
    test_masks = masks[atlas_size:]

    out_root.mkdir(parents=True, exist_ok=True)

    scores = []

    for t in range(len(test_images)):
        fixed_img = test_images[t]
        gt_mask = test_masks[t]

        case_dir = ensure_clean_dir(out_root / f"test_{t:03d}")
        warped_masks = []

        fixed_np = sitk.GetArrayFromImage(fixed_img)
        slice_idx = fixed_np.shape[0] // 2

        for a in range(atlas_size):
            pair_dir = case_dir / f"atlas_{a:03d}_to_test"
            final_tp = register_atlas_to_fixed(fixed_img, atlas_images[a], pair_dir)
            warped = warp_label_with_transformix(atlas_masks[a], final_tp, pair_dir / "mask_warp")
            warped_masks.append(warped)

        fused = majority_vote(warped_masks)
        sitk.WriteImage(fused, str(case_dir / "fused_mask_majority_vote.nii.gz"))
        sitk.WriteImage(sitk.Cast(gt_mask > 0, sitk.sitkUInt8), str(case_dir / "gt_mask.nii.gz"))

        d = dice_score(fused, gt_mask)
        scores.append((t, d))

        # save a preview image instead of blocking plt.show()
        fused_np = sitk.GetArrayFromImage(fused)
        gt_np = sitk.GetArrayFromImage(sitk.Cast(gt_mask > 0, sitk.sitkUInt8))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(fixed_np[slice_idx], cmap="gray")
        plt.title("Fixed (test)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(fixed_np[slice_idx], cmap="gray")
        plt.imshow(np.ma.masked_where(fused_np[slice_idx] == 0, fused_np[slice_idx]), cmap="Reds", alpha=0.4)
        plt.title(f"Fused mask (Dice={d:.3f})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(fixed_np[slice_idx], cmap="gray")
        plt.imshow(np.ma.masked_where(gt_np[slice_idx] == 0, gt_np[slice_idx]), cmap="Reds", alpha=0.4)
        plt.title("GT mask")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(str(case_dir / "preview.png"), dpi=150, bbox_inches="tight")
        plt.close()

        (case_dir / "summary.txt").write_text(
            "Atlas-based segmentation with non-linear registration\n"
            f"atlas_size = {atlas_size}\n"
            f"dice = {d:.6f}\n",
            encoding="utf-8",
        )

        print(f"test_{t:03d}: Dice={d:.4f}")

    # save experiment scores
    csv_path = out_root / "scores.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("test_index,dice\n")
        for t, d in scores:
            f.write(f"{t},{d:.6f}\n")

    dice_vals = np.array([d for _, d in scores], dtype=np.float32)
    (out_root / "scores_summary.txt").write_text(
        "Atlas-based segmentation (non-linear)\n"
        f"atlas_size = {atlas_size}\n"
        f"n_test = {len(scores)}\n"
        f"mean_dice = {float(dice_vals.mean()):.6f}\n"
        f"std_dice  = {float(dice_vals.std()):.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    ROOT = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train")
    OUT_ROOT = Path(r"C:\temp\atlas_segmentation_nonrigid_full")

    run_full_experiment(ROOT, OUT_ROOT, atlas_size=8)