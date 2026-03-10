from pathlib import Path
import time
import math
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from prostateLoader import ProstateLoader


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(r"C:\Users\30697\OneDrive - University of West Attica\Documents\GitHub\CSinMIA\prostate158_train\train")
AFFINE_PARAM = Path(r"C:\Users\30697\OneDrive - University of West Attica\Documents\GitHub\CSinMIA\ParameterFiles\Affine\affine.txt")
BSPLINE_PARAM = Path(r"C:\Users\30697\OneDrive - University of West Attica\Documents\GitHub\CSinMIA\ParameterFiles\BSpline\bspline.txt")
OUT_ROOT = Path(r"C:\temp\nonlinear_registration_test")


# ------------------------------------------------------------
# Settings, same style as his script
# ------------------------------------------------------------
ATLAS_SIZE = 50
PRESELECTION_SIZE = 5
VISUALISATION_SLICE = 15
TEST_IMAGE_LOCAL_INDEX = 1


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_tp(out_dir: Path) -> Path:
    tps = sorted(out_dir.glob("TransformParameters.*.txt"))
    if not tps:
        raise FileNotFoundError(f"No TransformParameters.*.txt found in {out_dir}")
    return tps[-1]


def binarize(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def dice_score(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    return 1.0 if den == 0 else 2.0 * inter / den


def jaccard_score(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    union = int(((p | g) > 0).sum())
    return 1.0 if union == 0 else inter / union


def hausdorff_distance_mm(pred: sitk.Image, gt: sitk.Image) -> float:
    pred_b = binarize(pred)
    gt_b = binarize(gt)

    pred_sum = int(sitk.GetArrayFromImage(pred_b).sum())
    gt_sum = int(sitk.GetArrayFromImage(gt_b).sum())

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return np.inf

    hd = sitk.HausdorffDistanceImageFilter()
    hd.Execute(pred_b, gt_b)
    return float(hd.GetHausdorffDistance())


def normalized_cross_correlation(fixed_img: sitk.Image, registered_img: sitk.Image) -> float:
    f = sitk.GetArrayFromImage(fixed_img).astype(np.float32)
    r = sitk.GetArrayFromImage(registered_img).astype(np.float32)

    mask = (f != 0) | (r != 0)
    if mask.sum() < 50:
        return -1.0

    f = f[mask]
    r = r[mask]

    f_std = float(f.std())
    r_std = float(r.std())
    if f_std < 1e-8 or r_std < 1e-8:
        return -1.0

    f = (f - f.mean()) / f_std
    r = (r - r.mean()) / r_std
    return float(np.mean(f * r))


def vote_fusion(masks: list[sitk.Image]) -> sitk.Image:
    arrs = [sitk.GetArrayFromImage(binarize(m)).astype(np.uint8) for m in masks]
    stack = np.stack(arrs, axis=0)
    votes = stack.sum(axis=0)

    threshold = math.ceil(len(masks) / 2)
    fused = (votes >= threshold).astype(np.uint8)

    fused_img = sitk.GetImageFromArray(fused)
    fused_img.CopyInformation(masks[0])
    return fused_img


# ------------------------------------------------------------
# Registration
# ------------------------------------------------------------
def register_affine_then_bspline(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    out_dir: Path,
) -> tuple[sitk.Image, Path]:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "affine_stage")
    ensure_dir(out_dir / "bspline_stage")

    # Affine stage
    elx1 = sitk.ElastixImageFilter()
    elx1.SetFixedImage(fixed_img)
    elx1.SetMovingImage(moving_img)
    elx1.SetOutputDirectory(str(out_dir / "affine_stage"))
    elx1.LogToConsoleOff()
    elx1.LogToFileOn()
    elx1.SetLogFileName(str(out_dir / "affine_stage" / "elastix.log"))
    elx1.SetParameterMap(sitk.ReadParameterFile(str(AFFINE_PARAM)))
    elx1.Execute()

    affine_tp = latest_tp(out_dir / "affine_stage")

    # Nonlinear Bspline stage
    bspline_pm = sitk.ReadParameterFile(str(BSPLINE_PARAM))
    bspline_pm["InitialTransformParameterFileName"] = [str(affine_tp).replace("\\", "/")]
    bspline_pm["HowToCombineTransforms"] = ["Compose"]

    elx2 = sitk.ElastixImageFilter()
    elx2.SetFixedImage(fixed_img)
    elx2.SetMovingImage(moving_img)
    elx2.SetOutputDirectory(str(out_dir / "bspline_stage"))
    elx2.LogToConsoleOff()
    elx2.LogToFileOn()
    elx2.SetLogFileName(str(out_dir / "bspline_stage" / "elastix.log"))
    elx2.SetParameterMap(bspline_pm)
    elx2.Execute()

    registered = elx2.GetResultImage()
    sitk.WriteImage(registered, str(out_dir / "registered.nii.gz"))

    final_tp = latest_tp(out_dir / "bspline_stage")
    return registered, final_tp


def warp_mask(mask_img: sitk.Image, final_tp: Path, out_dir: Path) -> sitk.Image:
    ensure_dir(out_dir)

    tp_text = final_tp.read_text(encoding="utf-8")

    if '(ResampleInterpolator "FinalBSplineInterpolator")' in tp_text:
        tp_text = tp_text.replace(
            '(ResampleInterpolator "FinalBSplineInterpolator")',
            '(ResampleInterpolator "FinalNearestNeighborInterpolator")'
        )
    elif '(ResampleInterpolator "FinalLinearInterpolator")' in tp_text:
        tp_text = tp_text.replace(
            '(ResampleInterpolator "FinalLinearInterpolator")',
            '(ResampleInterpolator "FinalNearestNeighborInterpolator")'
        )

    tp_nn = out_dir / "TransformParameters.NN.txt"
    tp_nn.write_text(tp_text, encoding="utf-8")

    tfx = sitk.TransformixImageFilter()
    tfx.SetMovingImage(mask_img)
    tfx.SetTransformParameterMap(sitk.ReadParameterFile(str(tp_nn)))
    tfx.SetOutputDirectory(str(out_dir))
    tfx.LogToConsoleOff()
    tfx.LogToFileOff()
    tfx.Execute()

    warped = tfx.GetResultImage()
    warped = sitk.Cast(warped > 0.5, sitk.sitkUInt8)
    sitk.WriteImage(warped, str(out_dir / "warped_mask.nii.gz"))
    return warped


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    t0 = time.perf_counter()

    ensure_dir(OUT_ROOT)

    loader = ProstateLoader(str(ROOT))
    images, segmen = loader.LoadData()

    atlas_images = images[0:ATLAS_SIZE]
    atlas_segmen = segmen[0:ATLAS_SIZE]

    test_images = images[ATLAS_SIZE:]
    test_segmen = segmen[ATLAS_SIZE:]

    t1 = time.perf_counter()

    fixed_img = test_images[TEST_IMAGE_LOCAL_INDEX]
    gt_mask = test_segmen[TEST_IMAGE_LOCAL_INDEX]
    global_test_idx = ATLAS_SIZE + TEST_IMAGE_LOCAL_INDEX

    print(f"Using test image global index: {global_test_idx}")

    metrics = []
    reg_results = []
    reg_segmen = []
    atlas_indices = []

    for idx, moving_img in enumerate(atlas_images):
        print(f"Registering atlas {idx:03d} -> test {global_test_idx:03d}")

        pair_dir = ensure_dir(OUT_ROOT / f"atlas_{idx:03d}_to_test_{global_test_idx:03d}")

        try:
            registered_img, final_tp = register_affine_then_bspline(
                fixed_img=fixed_img,
                moving_img=moving_img,
                out_dir=pair_dir,
            )

            warped_mask = warp_mask(
                mask_img=atlas_segmen[idx],
                final_tp=final_tp,
                out_dir=pair_dir / "mask_warp",
            )

            metric = normalized_cross_correlation(fixed_img, registered_img)

            metrics.append(metric)
            reg_results.append(registered_img)
            reg_segmen.append(warped_mask)
            atlas_indices.append(idx)

        except Exception as e:
            print(f"atlas {idx:03d} failed: {e}")
            with (pair_dir / "FAILED.txt").open("w", encoding="utf-8") as f:
                f.write(str(e))

    t2 = time.perf_counter()

    if len(metrics) == 0:
        print("No successful registrations.")
        return

    results = list(zip(metrics, atlas_indices, reg_results, reg_segmen))
    results_sorted = sorted(results, key=lambda t: t[0], reverse=True)

    top_results = results_sorted[:PRESELECTION_SIZE]

    top_metrics = [r[0] for r in top_results]
    top_indices = [r[1] for r in top_results]
    top_reg = [r[2] for r in top_results]
    top_masks = [r[3] for r in top_results]

    t3 = time.perf_counter()

    fused = vote_fusion(top_masks)
    sitk.WriteImage(fused, str(OUT_ROOT / "fused_mask_topk.nii.gz"))
    sitk.WriteImage(binarize(gt_mask), str(OUT_ROOT / "gt_mask.nii.gz"))

    dice = dice_score(fused, gt_mask)
    jacc = jaccard_score(fused, gt_mask)
    hd = hausdorff_distance_mm(fused, gt_mask)

    with (OUT_ROOT / "atlas_ranking.csv").open("w", encoding="utf-8") as f:
        f.write("rank,atlas_idx,ncc\n")
        for rank, row in enumerate(results_sorted, start=1):
            f.write(f"{rank},{row[1]},{row[0]:.6f}\n")

    with (OUT_ROOT / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"test_image_global_index = {global_test_idx}\n")
        f.write(f"atlas_size = {ATLAS_SIZE}\n")
        f.write(f"preselection_size = {PRESELECTION_SIZE}\n")
        f.write(f"dice = {dice:.6f}\n")
        f.write(f"jaccard = {jacc:.6f}\n")
        f.write(f"hausdorff_mm = {hd}\n")
        f.write(f"selected_atlases = {top_indices}\n")
        f.write(f"selected_ncc = {[round(x, 6) for x in top_metrics]}\n")

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.imshow(fixed_img[:, :, VISUALISATION_SLICE], cmap="gray")
    plt.title("Test image")

    for i, img in enumerate(top_reg):
        plt.subplot(2, 3, i + 2)
        plt.imshow(img[:, :, VISUALISATION_SLICE], cmap="gray")
        plt.title(f"NCC: {top_metrics[i]:.3f}")

    plt.tight_layout()
    plt.savefig(str(OUT_ROOT / "top_registered_images.png"), dpi=150, bbox_inches="tight")

    fixed_np = sitk.GetArrayFromImage(fixed_img)
    fused_np = sitk.GetArrayFromImage(binarize(fused))
    gt_np = sitk.GetArrayFromImage(binarize(gt_mask))
    slice_idx = min(VISUALISATION_SLICE, fixed_np.shape[2] - 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(fixed_np[:, :, slice_idx], cmap="gray")
    plt.title(f"Fixed test {global_test_idx:03d}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(fixed_np[:, :, slice_idx], cmap="gray")
    plt.imshow(np.ma.masked_where(fused_np[:, :, slice_idx] == 0, fused_np[:, :, slice_idx]), cmap="Reds", alpha=0.4)
    plt.title(f"Fused top-{PRESELECTION_SIZE}, Dice={dice:.3f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fixed_np[:, :, slice_idx], cmap="gray")
    plt.imshow(np.ma.masked_where(gt_np[:, :, slice_idx] == 0, gt_np[:, :, slice_idx]), cmap="Reds", alpha=0.4)
    plt.title("GT mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(str(OUT_ROOT / "preview.png"), dpi=150, bbox_inches="tight")

    t4 = time.perf_counter()

    print("___________________________________________")
    print(f"Loading the images      : {(t1 - t0)//60:.0f}m {(t1 - t0)%60:.2f}s")
    print(f"Affine + Bspline regs   : {(t2 - t1)//60:.0f}m {(t2 - t1)%60:.2f}s")
    print(f"Preselection            : {(t3 - t2)//60:.0f}m {(t3 - t2)%60:.2f}s")
    print(f"Plotting + saving       : {(t4 - t3)//60:.0f}m {(t4 - t3)%60:.2f}s")
    print("___________________________________________")
    print(f"Total time              : {(t4 - t0)//60:.0f}m {(t4 - t0)%60:.2f}s")
    print()
    print(f"Dice = {dice:.4f}")
    print(f"Jaccard = {jacc:.4f}")
    print(f"Hausdorff = {hd}")
    print(f"Selected atlases = {top_indices}")


if __name__ == "__main__":
    main()