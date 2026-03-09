from pathlib import Path
from datetime import datetime
import math
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from prostateLoader import ProstateLoader


# ============================================================
# Configuration
# ============================================================
ROOT = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train")
BASE_OUT_ROOT = Path(r"C:\temp\atlas_segmentation_multistep_runs")

SUCCESS_DICE_THRESHOLD = 0.70

FIRST4_ATLASES = [0, 1, 2, 3]
FIRST4_POSTREG_THRESHOLD = 0.40

PRIMARY_ATLAS_INDICES = [135, 10, 28, 127, 87, 14, 35, 129]
PARTNER_MAP = {
    135: [73, 102],
    10: [105, 11],
    28: [83, 104],
    127: [39, 95],
    87: [111, 91],
    14: [106, 96],
    35: [48, 2],
    129: [133, 109],
}
PARTNER_THRESHOLD = 0.35

TARGET_SIZE = (64, 64, 32)
TOP_K_SIMILAR = 2
SIMILARITY_POSTREG_THRESHOLD = 0.40

MAX_TEST_CASES = None
VOTE_THRESHOLD = None


# ============================================================
# Utilities
# ============================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_elastix_path(p: Path) -> str:
    return str(p).replace("\\", "/")


def latest_tp(out_dir: Path) -> Path:
    tps = sorted(out_dir.glob("TransformParameters.*.txt"))
    if not tps:
        raise FileNotFoundError(f"No TransformParameters.*.txt found in: {out_dir}")
    return tps[-1]


def create_unique_run_folder(base_out_root: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out_root / f"{label}_{timestamp}"

    counter = 1
    while run_dir.exists():
        run_dir = base_out_root / f"{label}_{timestamp}_{counter:02d}"
        counter += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving)


def resample_to_size(img: sitk.Image, out_size=(64, 64, 32), is_label=False) -> sitk.Image:
    original_size = np.array(list(img.GetSize()), dtype=np.int32)
    original_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)

    out_size = np.array(out_size, dtype=np.int32)
    out_spacing = original_spacing * (original_size / out_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(x) for x in out_size])
    resampler.SetOutputSpacing([float(x) for x in out_spacing])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(img)


def zscore_nonzero(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mask = arr != 0
    if mask.sum() == 0:
        return arr

    vals = arr[mask]
    mean = float(vals.mean())
    std = float(vals.std())

    out = np.zeros_like(arr, dtype=np.float32)
    if std < 1e-8:
        out[mask] = vals - mean
    else:
        out[mask] = (vals - mean) / std
    return out


def volume_to_feature(img: sitk.Image, target_size=(64, 64, 32)) -> np.ndarray:
    small = resample_to_size(img, out_size=target_size, is_label=False)
    arr = sitk.GetArrayFromImage(small)
    arr = zscore_nonzero(arr)
    return arr.ravel()


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_std = float(a.std())
    b_std = float(b.std())

    if a_std < 1e-8 or b_std < 1e-8:
        return 1.0

    a_n = (a - a.mean()) / a_std
    b_n = (b - b.mean()) / b_std
    corr = float(np.mean(a_n * b_n))
    corr = max(-1.0, min(1.0, corr))
    return 1.0 - corr


def compute_pairwise_distance_matrix(features: list[np.ndarray]) -> np.ndarray:
    n = len(features)
    D = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            d = correlation_distance(features[i], features[j])
            D[i, j] = d
            D[j, i] = d

    return D


# ============================================================
# Metrics
# ============================================================
def binarize(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def dice_score(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    den = int(p.sum() + g.sum())
    return 1.0 if den == 0 else (2.0 * inter / den)


def jaccard_score(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    inter = int((p & g).sum())
    union = int(((p | g) > 0).sum())
    return 1.0 if union == 0 else (inter / union)


def relative_volume_difference(pred: sitk.Image, gt: sitk.Image) -> float:
    p = sitk.GetArrayFromImage(binarize(pred)).astype(np.uint8)
    g = sitk.GetArrayFromImage(binarize(gt)).astype(np.uint8)
    vp = float(p.sum())
    vg = float(g.sum())
    if vg == 0:
        return 0.0 if vp == 0 else np.inf
    return (vp - vg) / vg


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


# ============================================================
# Elastix parameter maps
# ============================================================
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
    pm["GridSpacingSchedule"] = [v for s in schedule_1d for v in (s, s, s)]

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


# ============================================================
# Registration and warping
# ============================================================
def register_atlas_to_fixed(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    out_dir: Path,
) -> tuple[Path, sitk.Image]:
    ensure_dir(out_dir)
    log_dir = ensure_dir(out_dir / "_logs")
    pm_dir = ensure_dir(out_dir / "_param_files")

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

    return final_tp_copy, registered


def warp_label_with_transformix(label_img: sitk.Image, final_tp_file: Path, out_dir: Path) -> sitk.Image:
    ensure_dir(out_dir)

    tp_text = final_tp_file.read_text(encoding="utf-8")

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


def run_one_atlas_registration(
    atlas_case_idx: int,
    fixed_case_idx: int,
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    moving_mask: sitk.Image,
    case_dir: Path,
) -> dict | None:
    pair_dir = ensure_dir(case_dir / f"atlas_{atlas_case_idx:03d}_to_test_{fixed_case_idx:03d}")

    try:
        final_tp, registered_img = register_atlas_to_fixed(
            fixed_img=fixed_img,
            moving_img=moving_img,
            out_dir=pair_dir,
        )

        warped_mask = warp_label_with_transformix(
            label_img=moving_mask,
            final_tp_file=final_tp,
            out_dir=pair_dir / "mask_warp",
        )

        sim = normalized_cross_correlation(fixed_img, registered_img)

        return {
            "atlas_case_idx": atlas_case_idx,
            "similarity": sim,
            "registered_img": registered_img,
            "warped_mask": warped_mask,
        }

    except Exception as e:
        print(f"atlas_{atlas_case_idx:03d} failed for test_{fixed_case_idx:03d}: {e}")
        with (pair_dir / "FAILED.txt").open("w", encoding="utf-8") as f:
            f.write(str(e))
        return None


# ============================================================
# Fusion rules
# ============================================================
def choose_top2_by_threshold(results: list[dict | None], threshold: float) -> list[dict]:
    valid = [x for x in results if x is not None]
    if len(valid) == 0:
        return []

    valid = sorted(valid, key=lambda x: x["similarity"], reverse=True)
    above = [x for x in valid if x["similarity"] >= threshold]

    if len(above) >= 2:
        return above[:2]

    if len(above) == 1:
        return above

    return [valid[0]]


def vote_fusion(masks: list[sitk.Image], vote_threshold: int | None = None) -> sitk.Image:
    if len(masks) == 0:
        raise ValueError("No masks provided for fusion.")

    arrs = [sitk.GetArrayFromImage(binarize(m)).astype(np.uint8) for m in masks]
    stack = np.stack(arrs, axis=0)
    votes = stack.sum(axis=0)

    if vote_threshold is None:
        vote_threshold = math.ceil(len(masks) / 2)

    fused = (votes >= vote_threshold).astype(np.uint8)

    fused_img = sitk.GetImageFromArray(fused)
    fused_img.CopyInformation(masks[0])
    return fused_img


# ============================================================
# Output helpers
# ============================================================
def save_preview(case_dir: Path, fixed_img: sitk.Image, fused: sitk.Image, gt_mask: sitk.Image, title: str, dice: float) -> None:
    fixed_np = sitk.GetArrayFromImage(fixed_img)
    fused_np = sitk.GetArrayFromImage(binarize(fused))
    gt_np = sitk.GetArrayFromImage(binarize(gt_mask))
    slice_idx = fixed_np.shape[0] // 2

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(fixed_np[slice_idx], cmap="gray")
    plt.title("Fixed")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(fixed_np[slice_idx], cmap="gray")
    plt.imshow(np.ma.masked_where(fused_np[slice_idx] == 0, fused_np[slice_idx]), cmap="Reds", alpha=0.4)
    plt.title(f"{title}, Dice={dice:.3f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(fixed_np[slice_idx], cmap="gray")
    plt.imshow(np.ma.masked_where(gt_np[slice_idx] == 0, gt_np[slice_idx]), cmap="Reds", alpha=0.4)
    plt.title("GT")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(str(case_dir / "preview.png"), dpi=150, bbox_inches="tight")
    plt.close()


def package_method_result(
    method_name: str,
    method_dir: Path,
    fused: sitk.Image,
    gt_mask: sitk.Image,
    fixed_img: sitk.Image,
    chosen_results: list[dict],
    vote_threshold: int | None,
    extra_lines: list[str],
) -> dict:
    sitk.WriteImage(fused, str(method_dir / "fused_mask.nii.gz"))
    sitk.WriteImage(binarize(gt_mask), str(method_dir / "gt_mask.nii.gz"))

    dice = dice_score(fused, gt_mask)
    jacc = jaccard_score(fused, gt_mask)
    hd = hausdorff_distance_mm(fused, gt_mask)
    rvd = relative_volume_difference(fused, gt_mask)

    effective_vote_threshold = vote_threshold if vote_threshold is not None else math.ceil(len(chosen_results) / 2)

    with (method_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"method = {method_name}\n")
        f.write(f"n_selected_masks = {len(chosen_results)}\n")
        f.write(f"vote_threshold = {effective_vote_threshold}\n")
        f.write(f"dice = {dice:.6f}\n")
        f.write(f"jaccard = {jacc:.6f}\n")
        f.write(f"hausdorff_mm = {hd}\n")
        f.write(f"rvd = {rvd:.6f}\n")
        for line in extra_lines:
            f.write(line + "\n")

    save_preview(method_dir, fixed_img, fused, gt_mask, method_name, dice)

    return {
        "method_name": method_name,
        "method_dir": method_dir,
        "dice": dice,
        "jaccard": jacc,
        "hausdorff_mm": hd,
        "rvd": rvd,
        "chosen_results": chosen_results,
        "fused": fused,
    }


# ============================================================
# Method 1: first 4 fixed atlases
# ============================================================
def run_method1_first4(
    case_idx: int,
    images: list[sitk.Image],
    masks: list[sitk.Image],
    patient_dir: Path,
    vote_threshold: int | None,
) -> dict | None:
    if case_idx in FIRST4_ATLASES:
        return None

    method_dir = ensure_dir(patient_dir / "method_1_first4")

    fixed_img = images[case_idx]
    gt_mask = masks[case_idx]

    raw_results = []
    for atlas_idx in FIRST4_ATLASES:
        r = run_one_atlas_registration(
            atlas_case_idx=atlas_idx,
            fixed_case_idx=case_idx,
            fixed_img=fixed_img,
            moving_img=images[atlas_idx],
            moving_mask=masks[atlas_idx],
            case_dir=method_dir,
        )
        raw_results.append(r)

    chosen_results = choose_top2_by_threshold(raw_results, threshold=FIRST4_POSTREG_THRESHOLD)
    if len(chosen_results) == 0:
        return None

    fused = vote_fusion([x["warped_mask"] for x in chosen_results], vote_threshold=vote_threshold)

    with (method_dir / "atlas_ranking.csv").open("w", encoding="utf-8") as f:
        f.write("rank,atlas_case_idx,postreg_similarity,chosen\n")
        sortable = []
        for atlas_idx, r in zip(FIRST4_ATLASES, raw_results):
            sim = float("-inf") if r is None else float(r["similarity"])
            chosen = 0 if r is None else int(any(c["atlas_case_idx"] == atlas_idx for c in chosen_results))
            sortable.append((atlas_idx, sim, chosen))
        sortable.sort(key=lambda x: x[1], reverse=True)
        for rank, row in enumerate(sortable, start=1):
            f.write(f"{rank},{row[0]},{row[1]:.6f},{row[2]}\n")

    extra_lines = [
        f"atlases = {FIRST4_ATLASES}",
        f"postreg_threshold = {FIRST4_POSTREG_THRESHOLD}",
        f"chosen_atlases = {[int(x['atlas_case_idx']) for x in chosen_results]}",
    ]

    return package_method_result(
        method_name="method_1_first4",
        method_dir=method_dir,
        fused=fused,
        gt_mask=gt_mask,
        fixed_img=fixed_img,
        chosen_results=chosen_results,
        vote_threshold=vote_threshold,
        extra_lines=extra_lines,
    )


# ============================================================
# Method 2: diverse family atlas method
# ============================================================
def run_method2_family(
    case_idx: int,
    images: list[sitk.Image],
    masks: list[sitk.Image],
    patient_dir: Path,
    vote_threshold: int | None,
) -> dict | None:
    reserved_cases = set(PRIMARY_ATLAS_INDICES)
    for p in PRIMARY_ATLAS_INDICES:
        reserved_cases.update(PARTNER_MAP[p])

    if case_idx in reserved_cases:
        return None

    method_dir = ensure_dir(patient_dir / "method_2_family")
    fixed_img = images[case_idx]
    gt_mask = masks[case_idx]

    primary_results = []
    for atlas_idx in PRIMARY_ATLAS_INDICES:
        r = run_one_atlas_registration(
            atlas_case_idx=atlas_idx,
            fixed_case_idx=case_idx,
            fixed_img=fixed_img,
            moving_img=images[atlas_idx],
            moving_mask=masks[atlas_idx],
            case_dir=method_dir,
        )
        if r is not None:
            primary_results.append(r)

    valid_primary = [x for x in primary_results if x["similarity"] >= 0.0]
    if len(valid_primary) == 0:
        return None

    valid_primary.sort(key=lambda x: x["similarity"], reverse=True)
    best_primary = valid_primary[0]

    chosen_results = [best_primary]
    family_lines = [
        f"best_primary = {best_primary['atlas_case_idx']}",
        f"best_primary_similarity = {best_primary['similarity']:.6f}",
        f"partner_threshold = {PARTNER_THRESHOLD}",
    ]

    for partner_idx in PARTNER_MAP[best_primary["atlas_case_idx"]]:
        r = run_one_atlas_registration(
            atlas_case_idx=partner_idx,
            fixed_case_idx=case_idx,
            fixed_img=fixed_img,
            moving_img=images[partner_idx],
            moving_mask=masks[partner_idx],
            case_dir=method_dir,
        )

        if r is None:
            family_lines.append(f"partner_{partner_idx} = failed")
            continue

        family_lines.append(f"partner_{partner_idx}_similarity = {r['similarity']:.6f}")
        if r["similarity"] >= PARTNER_THRESHOLD:
            chosen_results.append(r)
            family_lines.append(f"partner_{partner_idx} = included")
        else:
            family_lines.append(f"partner_{partner_idx} = rejected")

    fused = vote_fusion([x["warped_mask"] for x in chosen_results], vote_threshold=vote_threshold)

    with (method_dir / "primary_atlas_ranking.csv").open("w", encoding="utf-8") as f:
        f.write("rank,atlas_case_idx,postreg_similarity,is_best_primary\n")
        primary_sorted = sorted(primary_results, key=lambda x: x["similarity"], reverse=True)
        for rank, item in enumerate(primary_sorted, start=1):
            f.write(f"{rank},{item['atlas_case_idx']},{item['similarity']:.6f},{int(item['atlas_case_idx'] == best_primary['atlas_case_idx'])}\n")

    return package_method_result(
        method_name="method_2_family",
        method_dir=method_dir,
        fused=fused,
        gt_mask=gt_mask,
        fixed_img=fixed_img,
        chosen_results=chosen_results,
        vote_threshold=vote_threshold,
        extra_lines=family_lines + [f"chosen_atlases = {[int(x['atlas_case_idx']) for x in chosen_results]}"],
    )


# ============================================================
# Method 3: patient-specific top-2 similarity
# ============================================================
def run_method3_similarity_top2(
    case_idx: int,
    images: list[sitk.Image],
    masks: list[sitk.Image],
    D: np.ndarray,
    patient_dir: Path,
    vote_threshold: int | None,
) -> dict | None:
    method_dir = ensure_dir(patient_dir / "method_3_similarity_top2")
    fixed_img = images[case_idx]
    gt_mask = masks[case_idx]

    distances = D[case_idx].copy()
    distances[case_idx] = np.inf
    candidate_indices = np.argsort(distances)[:TOP_K_SIMILAR].tolist()

    raw_results = []
    for atlas_idx in candidate_indices:
        r = run_one_atlas_registration(
            atlas_case_idx=atlas_idx,
            fixed_case_idx=case_idx,
            fixed_img=fixed_img,
            moving_img=images[atlas_idx],
            moving_mask=masks[atlas_idx],
            case_dir=method_dir,
        )
        raw_results.append(r)

    chosen_results = choose_top2_by_threshold(raw_results, threshold=SIMILARITY_POSTREG_THRESHOLD)
    if len(chosen_results) == 0:
        return None

    fused = vote_fusion([x["warped_mask"] for x in chosen_results], vote_threshold=vote_threshold)

    with (method_dir / "atlas_ranking.csv").open("w", encoding="utf-8") as f:
        f.write("rank,atlas_case_idx,preselection_distance,postreg_similarity,chosen\n")
        sortable = []
        for atlas_idx, r in zip(candidate_indices, raw_results):
            sim = float("-inf") if r is None else float(r["similarity"])
            chosen = 0 if r is None else int(any(c["atlas_case_idx"] == atlas_idx for c in chosen_results))
            sortable.append((atlas_idx, float(D[case_idx, atlas_idx]), sim, chosen))
        sortable.sort(key=lambda x: x[2], reverse=True)
        for rank, row in enumerate(sortable, start=1):
            f.write(f"{rank},{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]}\n")

    extra_lines = [
        f"preselected_atlases = {candidate_indices}",
        f"postreg_threshold = {SIMILARITY_POSTREG_THRESHOLD}",
        f"chosen_atlases = {[int(x['atlas_case_idx']) for x in chosen_results]}",
    ]
    for atlas_idx in candidate_indices:
        extra_lines.append(f"preselection_distance_to_{atlas_idx} = {float(D[case_idx, atlas_idx]):.6f}")

    return package_method_result(
        method_name="method_3_similarity_top2",
        method_dir=method_dir,
        fused=fused,
        gt_mask=gt_mask,
        fixed_img=fixed_img,
        chosen_results=chosen_results,
        vote_threshold=vote_threshold,
        extra_lines=extra_lines,
    )


# ============================================================
# Master pipeline
# ============================================================
def run_multistep_pipeline(
    root: Path,
    base_out_root: Path,
    max_test_cases: int | None = None,
    vote_threshold: int | None = None,
    success_dice_threshold: float = 0.70,
) -> None:
    loader = ProstateLoader(str(root))
    images, masks = loader.LoadData()
    total_cases = len(images)

    out_root = create_unique_run_folder(base_out_root, "multistep_atlas_pipeline")

    print(f"\nResults will be saved in:\n{out_root}\n")
    print(f"Loaded {total_cases} cases")
    print("Building similarity features once for Method 3...")

    features = []
    for i, img in enumerate(images):
        features.append(volume_to_feature(img, target_size=TARGET_SIZE))
        print(f"  processed case {i:03d}")

    print("Computing pairwise image-distance matrix...")
    D = compute_pairwise_distance_matrix(features)

    all_case_indices = list(range(total_cases))
    if max_test_cases is not None:
        if max_test_cases < 1:
            raise ValueError("max_test_cases must be at least 1 or None.")
        all_case_indices = all_case_indices[:max_test_cases]

    (out_root / "run_settings.txt").write_text(
        "Multi-step offline atlas segmentation cascade\n"
        f"dataset_root = {root}\n"
        f"total_loaded_cases = {total_cases}\n"
        f"success_dice_threshold = {success_dice_threshold}\n"
        f"method1_first4_atlases = {FIRST4_ATLASES}\n"
        f"method1_postreg_threshold = {FIRST4_POSTREG_THRESHOLD}\n"
        f"method2_primary_atlases = {PRIMARY_ATLAS_INDICES}\n"
        f"method2_partner_map = {PARTNER_MAP}\n"
        f"method2_partner_threshold = {PARTNER_THRESHOLD}\n"
        f"method3_target_size = {TARGET_SIZE}\n"
        f"method3_top_k_similar = {TOP_K_SIMILAR}\n"
        f"method3_postreg_threshold = {SIMILARITY_POSTREG_THRESHOLD}\n"
        f"num_cases_to_process = {len(all_case_indices)}\n"
        "important_note = method selection by Dice uses ground truth and is only valid for offline evaluation\n",
        encoding="utf-8",
    )

    global_rows = []

    for case_idx in all_case_indices:
        print(f"\n==================================================")
        print(f"Processing patient {case_idx:03d}")
        print(f"==================================================")

        patient_dir = ensure_dir(out_root / f"patient_{case_idx:03d}")
        tried_results = []

        method1_result = run_method1_first4(
            case_idx=case_idx,
            images=images,
            masks=masks,
            patient_dir=patient_dir,
            vote_threshold=vote_threshold,
        )
        if method1_result is not None:
            tried_results.append(method1_result)
            print(f"Method 1 Dice = {method1_result['dice']:.4f}")
            if method1_result["dice"] >= success_dice_threshold:
                best_result = method1_result
                final_method = method1_result["method_name"]
                final_reason = f"stopped after method 1, Dice >= {success_dice_threshold:.2f}"
            else:
                method2_result = run_method2_family(
                    case_idx=case_idx,
                    images=images,
                    masks=masks,
                    patient_dir=patient_dir,
                    vote_threshold=vote_threshold,
                )
                if method2_result is not None:
                    tried_results.append(method2_result)
                    print(f"Method 2 Dice = {method2_result['dice']:.4f}")
                    if method2_result["dice"] >= success_dice_threshold:
                        best_result = method2_result
                        final_method = method2_result["method_name"]
                        final_reason = f"stopped after method 2, Dice >= {success_dice_threshold:.2f}"
                    else:
                        method3_result = run_method3_similarity_top2(
                            case_idx=case_idx,
                            images=images,
                            masks=masks,
                            D=D,
                            patient_dir=patient_dir,
                            vote_threshold=vote_threshold,
                        )
                        if method3_result is not None:
                            tried_results.append(method3_result)
                            print(f"Method 3 Dice = {method3_result['dice']:.4f}")

                        if len(tried_results) == 0:
                            print("No valid method result for this patient.")
                            continue

                        best_result = max(tried_results, key=lambda x: x["dice"])
                        final_method = best_result["method_name"]
                        final_reason = "no method reached threshold, kept best Dice"
                else:
                    method3_result = run_method3_similarity_top2(
                        case_idx=case_idx,
                        images=images,
                        masks=masks,
                        D=D,
                        patient_dir=patient_dir,
                        vote_threshold=vote_threshold,
                    )
                    if method3_result is not None:
                        tried_results.append(method3_result)
                        print(f"Method 3 Dice = {method3_result['dice']:.4f}")

                    if len(tried_results) == 0:
                        print("No valid method result for this patient.")
                        continue

                    best_result = max(tried_results, key=lambda x: x["dice"])
                    final_method = best_result["method_name"]
                    final_reason = "method 2 unavailable or failed, kept best Dice"
        else:
            print("Method 1 skipped or failed.")
            method2_result = run_method2_family(
                case_idx=case_idx,
                images=images,
                masks=masks,
                patient_dir=patient_dir,
                vote_threshold=vote_threshold,
            )
            if method2_result is not None:
                tried_results.append(method2_result)
                print(f"Method 2 Dice = {method2_result['dice']:.4f}")
                if method2_result["dice"] >= success_dice_threshold:
                    best_result = method2_result
                    final_method = method2_result["method_name"]
                    final_reason = f"stopped after method 2, Dice >= {success_dice_threshold:.2f}"
                else:
                    method3_result = run_method3_similarity_top2(
                        case_idx=case_idx,
                        images=images,
                        masks=masks,
                        D=D,
                        patient_dir=patient_dir,
                        vote_threshold=vote_threshold,
                    )
                    if method3_result is not None:
                        tried_results.append(method3_result)
                        print(f"Method 3 Dice = {method3_result['dice']:.4f}")

                    if len(tried_results) == 0:
                        print("No valid method result for this patient.")
                        continue

                    best_result = max(tried_results, key=lambda x: x["dice"])
                    final_method = best_result["method_name"]
                    final_reason = "no method reached threshold, kept best Dice"
            else:
                method3_result = run_method3_similarity_top2(
                    case_idx=case_idx,
                    images=images,
                    masks=masks,
                    D=D,
                    patient_dir=patient_dir,
                    vote_threshold=vote_threshold,
                )
                if method3_result is not None:
                    tried_results.append(method3_result)
                    print(f"Method 3 Dice = {method3_result['dice']:.4f}")

                if len(tried_results) == 0:
                    print("No valid method result for this patient.")
                    continue

                best_result = max(tried_results, key=lambda x: x["dice"])
                final_method = best_result["method_name"]
                final_reason = "only method 3 available or valid"

        final_dir = ensure_dir(patient_dir / "final_selected_result")
        sitk.WriteImage(best_result["fused"], str(final_dir / "fused_mask_final.nii.gz"))
        sitk.WriteImage(binarize(masks[case_idx]), str(final_dir / "gt_mask.nii.gz"))

        with (patient_dir / "patient_summary.txt").open("w", encoding="utf-8") as f:
            f.write(f"patient_index = {case_idx}\n")
            f.write(f"final_method = {final_method}\n")
            f.write(f"final_reason = {final_reason}\n")
            f.write(f"final_dice = {best_result['dice']:.6f}\n")
            f.write(f"final_jaccard = {best_result['jaccard']:.6f}\n")
            f.write(f"final_hausdorff_mm = {best_result['hausdorff_mm']}\n")
            f.write(f"final_rvd = {best_result['rvd']:.6f}\n")
            f.write("\nmethods_tried:\n")
            for r in tried_results:
                f.write(f"  {r['method_name']}: dice={r['dice']:.6f}\n")

        save_preview(
            final_dir,
            images[case_idx],
            best_result["fused"],
            masks[case_idx],
            title=f"Final: {final_method}",
            dice=best_result["dice"],
        )

        global_rows.append(
            (
                case_idx,
                final_method,
                best_result["dice"],
                best_result["jaccard"],
                best_result["hausdorff_mm"],
                best_result["rvd"],
                final_reason,
            )
        )

        print(f"Final selected method for patient {case_idx:03d}: {final_method}, Dice={best_result['dice']:.4f}")

    if len(global_rows) == 0:
        print("\nFinished, but no valid patient results were produced.")
        print(f"Saved everything in:\n{out_root}")
        return

    with (out_root / "final_scores.csv").open("w", encoding="utf-8") as f:
        f.write("patient_index,final_method,dice,jaccard,hausdorff_mm,rvd,reason\n")
        for row in global_rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]},{row[5]:.6f},{row[6]}\n")

    dice_vals = np.array([x[2] for x in global_rows], dtype=np.float32)
    jacc_vals = np.array([x[3] for x in global_rows], dtype=np.float32)
    hd_vals = np.array([x[4] for x in global_rows], dtype=np.float64)
    rvd_vals = np.array([x[5] for x in global_rows], dtype=np.float32)
    finite_hd = hd_vals[np.isfinite(hd_vals)]

    (out_root / "final_scores_summary.txt").write_text(
        "Final selected results across all patients\n"
        f"n_cases = {len(global_rows)}\n"
        f"mean_dice = {float(dice_vals.mean()):.6f}\n"
        f"std_dice = {float(dice_vals.std()):.6f}\n"
        f"mean_jaccard = {float(jacc_vals.mean()):.6f}\n"
        f"std_jaccard = {float(jacc_vals.std()):.6f}\n"
        f"mean_rvd = {float(rvd_vals.mean()):.6f}\n"
        f"std_rvd = {float(rvd_vals.std()):.6f}\n"
        f"mean_hausdorff_mm = {float(finite_hd.mean()) if len(finite_hd) > 0 else float('nan')}\n"
        f"std_hausdorff_mm = {float(finite_hd.std()) if len(finite_hd) > 0 else float('nan')}\n",
        encoding="utf-8",
    )

    print("\nFinished.")
    print(f"Saved everything in:\n{out_root}")


if __name__ == "__main__":
    run_multistep_pipeline(
        root=ROOT,
        base_out_root=BASE_OUT_ROOT,
        max_test_cases=MAX_TEST_CASES,
        vote_threshold=VOTE_THRESHOLD,
        success_dice_threshold=SUCCESS_DICE_THRESHOLD,
    )