from pathlib import Path
import numpy as np
import SimpleITK as sitk

from prostateLoader import ProstateLoader


def ncc(a, b, eps=1e-8):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + eps)
    b = (b - b.mean()) / (b.std() + eps)
    return float((a * b).mean())


def dice(a, b, eps=1e-8):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return float((2.0 * inter) / (a.sum() + b.sum() + eps))


def warp_mask_nearest(moving_mask: sitk.Image, transform_parameter_map):
    tmap = transform_parameter_map
    for m in tmap:
        m["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        m["FinalBSplineInterpolationOrder"] = ["0"]
        m["ResultImagePixelType"] = ["unsigned char"]

    tx = sitk.TransformixImageFilter()
    tx.SetTransformParameterMap(tmap)
    tx.SetMovingImage(moving_mask)
    tx.Execute()

    out = tx.GetResultImage()
    out = sitk.Cast(out > 0, sitk.sitkUInt8)
    return out


PROJECT_ROOT = Path(r"C:\Users\30697\OneDrive - University of West Attica\Documents\GitHub\CSinMIA")
DATA_ROOT = PROJECT_ROOT / "prostate158_train" / "train"
LOG_ROOT = Path(r"C:\temp\elastix_logs_nonlinear_step1")

assert DATA_ROOT.exists(), f"Missing dataset folder: {DATA_ROOT}"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

if not hasattr(sitk, "ElastixImageFilter") or not hasattr(sitk, "TransformixImageFilter"):
    raise RuntimeError("No Elastix/Transformix in this SimpleITK build.")

loader = ProstateLoader(root=str(DATA_ROOT))
images, masks = loader.LoadData()

moving_index = 0
fixed_index = 8

fixed_img = images[fixed_index]
fixed_mask = masks[fixed_index]
moving_img = images[moving_index]
moving_mask = masks[moving_index]

run_dir = LOG_ROOT / f"moving_{moving_index:03d}_to_fixed_{fixed_index:03d}"
run_dir.mkdir(parents=True, exist_ok=True)

elx = sitk.ElastixImageFilter()
elx.SetFixedImage(fixed_img)
elx.SetMovingImage(moving_img)
elx.SetOutputDirectory(str(run_dir))
elx.LogToConsoleOn()
elx.LogToFileOn()

pm_translation = sitk.GetDefaultParameterMap("translation")
pm_affine = sitk.GetDefaultParameterMap("affine")
pm_bspline = sitk.GetDefaultParameterMap("bspline")

pm_affine["MaximumNumberOfIterations"] = ["128"]
pm_bspline["MaximumNumberOfIterations"] = ["256"]
pm_bspline["NumberOfResolutions"] = ["3"]
pm_bspline["FinalGridSpacingInPhysicalUnits"] = ["20.0", "20.0", "20.0"]

elx.SetParameterMap(pm_translation)
elx.AddParameterMap(pm_affine)
elx.AddParameterMap(pm_bspline)

elx.Execute()

reg_img = elx.GetResultImage()
tmap = elx.GetTransformParameterMap()

fixed_np = sitk.GetArrayFromImage(fixed_img)
reg_np = sitk.GetArrayFromImage(reg_img)
score_ncc = ncc(fixed_np, reg_np)

warped_mask = warp_mask_nearest(moving_mask, tmap)
warped_np = sitk.GetArrayFromImage(warped_mask)
fixed_mask_np = sitk.GetArrayFromImage(fixed_mask)
score_dice = dice(warped_np, fixed_mask_np)

print("Run folder:", run_dir)
print(f"NCC={score_ncc:.4f}  Dice={score_dice:.4f}")