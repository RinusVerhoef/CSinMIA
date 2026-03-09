from pathlib import Path
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
# Settings
# ------------------------------------------------------------
ATLAS_SIZE = 50
TEST_IMAGE_LOCAL_INDEX = 1
TOP_K = 5


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

    # Affine
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

    # Bspline
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
    ensure_dir(OUT_ROOT)

    loader = ProstateLoader(str(ROOT))
    images, masks = loader.LoadData()

    atlas_images = images[:ATLAS_SIZE]
    atlas_masks = masks[:ATLAS_SIZE]

    test_images = images[ATLAS_SIZE:]
    test_masks = masks[ATLAS_SIZE:]

    fixed_img = test_images[TEST_IMAGE_LOCAL_INDEX]
    gt_mask = test_masks[TEST_IMAGE_LOCAL_INDEX]
    global_test_idx = ATLAS_SIZE + TEST_IMAGE_LOCAL_INDEX

    print(f"Using test image global index: {global_test_idx}")

    atlas_results = []

    for atlas_idx in range(ATLAS_SIZE):
        print(f"Registering atlas {atlas_idx:03d} -> test {global_test_idx:03d}")

        pair_dir = ensure_dir(OUT_ROOT / f"atlas_{atlas_idx:03d}_to_test_{global_test_idx:03d}")

        try:
            registered_img, final_tp = register_affine_then_bspline(
                fixed_img=fixed_img,
                moving_img=atlas_images[atlas_idx],
                out_dir=pair_dir,
            )

            warped_mask = warp_mask(
                mask_img=atlas_masks[atlas_idx],
                final_tp=final_tp,
                out_dir=pair_dir / "mask_warp",
            )

            sim = normalized_cross_correlation(fixed_img, registered_img)

            atlas_results.append({
                "atlas_idx": atlas_idx,
                "similarity": sim,
                "registered_img": registered_img,
                "warped_mask": warped_mask,
            })

        except Exception as e:
            print(f"atlas {atlas_idx:03d} failed: {e}")
            with (pair_dir / "FAILED.txt").open("w", encoding="utf-8") as f:
                f.write(str(e))

    if len(atlas_results) == 0:
        print("No successful registrations.")
        return

    atlas_results.sort(key=lambda x: x["similarity"], reverse=True)

    with (OUT_ROOT / "atlas_ranking.csv").open("w", encoding="utf-8") as f:
        f.write("rank,atlas_idx,similarity\n")
        for rank, r in enumerate(atlas_results, start=1):
            f.write(f"{rank},{r['atlas_idx']},{r['similarity']:.6f}\n")

    top_results = atlas_results[:TOP_K]
    top_masks = [r["warped_mask"] for r in top_results]

    fused = vote_fusion(top_masks)
    sitk.WriteImage(fused, str(OUT_ROOT / "fused_mask_topk.nii.gz"))
    sitk.WriteImage(binarize(gt_mask), str(OUT_ROOT / "gt_mask.nii.gz"))

    dice = dice_score(fused, gt_mask)
    jacc = jaccard_score(fused, gt_mask)
    hd = hausdorff_distance_mm(fused, gt_mask)

    with (OUT_ROOT / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"test_image_global_index = {global_test_idx}\n")
        f.write(f"atlas_size = {ATLAS_SIZE}\n")
        f.write(f"top_k = {TOP_K}\n")
        f.write(f"dice = {dice:.6f}\n")
        f.write(f"jaccard = {jacc:.6f}\n")
        f.write(f"hausdorff_mm = {hd}\n")
        f.write("selected_atlases = " + ", ".join(str(r["atlas_idx"]) for r in top_results) + "\n")

    fixed_np = sitk.GetArrayFromImage(fixed_img)
    fused_np = sitk.GetArrayFromImage(binarize(fused))
    gt_np = sitk.GetArrayFromImage(binarize(gt_mask))
    slice_idx = fixed_np.shape[0] // 2

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(fixed_np[slice_idx], cmap="gray")
    plt.title(f"Fixed test {global_test_idx:03d}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(fixed_np[slice_idx], cmap="gray")
    plt.imshow(np.ma.masked_where(fused_np[slice_idx] == 0, fused_np[slice_idx]), cmap="Reds", alpha=0.4)
    plt.title(f"Fused top-{TOP_K}, Dice={dice:.3f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fixed_np[slice_idx], cmap="gray")
    plt.imshow(np.ma.masked_where(gt_np[slice_idx] == 0, gt_np[slice_idx]), cmap="Reds", alpha=0.4)
    plt.title("GT mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(str(OUT_ROOT / "preview.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Finished.")
    print(f"Dice = {dice:.4f}")
    print(f"Jaccard = {jacc:.4f}")
    print(f"Hausdorff = {hd}")


if __name__ == "__main__":
    main()