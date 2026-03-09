from pathlib import Path
import csv
import numpy as np
import SimpleITK as sitk

from prostateLoader import ProstateLoader


# ----------------------------
# Settings
# ----------------------------
ROOT = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train")
SELECTED_CASES = [135, 10, 28, 127, 87, 14, 35, 129]
TARGET_SIZE = (64, 64, 32)
TOP_MOST_SIMILAR = 2

OUT_CSV = Path(r"C:\temp\most_similar_to_selected.csv")
OUT_TXT = Path(r"C:\temp\most_similar_to_selected.txt")


# ----------------------------
# Utilities
# ----------------------------
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


# ----------------------------
# Main
# ----------------------------
def main():
    loader = ProstateLoader(str(ROOT))
    images, _ = loader.LoadData()

    n = len(images)
    print(f"Loaded {n} cases")

    print("Building low-resolution features...")
    features = []
    for i, img in enumerate(images):
        feat = volume_to_feature(img, target_size=TARGET_SIZE)
        features.append(feat)
        print(f"  processed case {i:03d}")

    print("Computing pairwise distance matrix...")
    D = compute_pairwise_distance_matrix(features)

    results = []

    print("\nMost similar cases for each selected patient:")
    for case_idx in SELECTED_CASES:
        distances = D[case_idx].copy()
        distances[case_idx] = np.inf

        nearest = np.argsort(distances)[:TOP_MOST_SIMILAR]

        print(f"\nCase {case_idx:03d}:")
        for rank, nn_idx in enumerate(nearest, start=1):
            d = float(D[case_idx, nn_idx])
            print(f"  {rank}. case {nn_idx:03d} | distance = {d:.4f}")
            results.append([case_idx, rank, int(nn_idx), f"{d:.4f}"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["selected_case", "neighbor_rank", "similar_case", "distance"])
        writer.writerows(results)

    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write("Most similar cases for each selected patient\n\n")
        current_case = None
        for row in results:
            case_idx, rank, nn_idx, d = row
            if case_idx != current_case:
                if current_case is not None:
                    f.write("\n")
                f.write(f"Case {case_idx:03d}:\n")
                current_case = case_idx
            f.write(f"  {rank}. case {nn_idx:03d} | distance = {d}\n")

    print(f"\nSaved CSV to: {OUT_CSV}")
    print(f"Saved TXT to: {OUT_TXT}")


if __name__ == "__main__":
    main()