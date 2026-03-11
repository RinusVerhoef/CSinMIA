from pathlib import Path
import SimpleITK as sitk


def _pm_get(pm, key):
    # sitk.ParameterMap behaves like a dict[str, list[str]]
    try:
        return pm.get(key)
    except Exception:
        return None


def run_registration(fixed_path: Path, moving_path: Path, out_dir: Path) -> None:
    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    out_dir = Path(out_dir)

    # -----------------------------
    # Hard checks (fail early)
    # -----------------------------
    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed image not found: {fixed_path}")
    if not moving_path.exists():
        raise FileNotFoundError(f"Moving image not found: {moving_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("RUNNING FILE:", Path(__file__).resolve())
    print("FIXED:", fixed_path.resolve())
    print("MOVING:", moving_path.resolve())
    print("OUT DIR:", out_dir.resolve())

    # -----------------------------
    # Read images
    # -----------------------------
    fixed = sitk.ReadImage(str(fixed_path))
    moving = sitk.ReadImage(str(moving_path))

    print("Fixed  size:", fixed.GetSize(), "spacing:", fixed.GetSpacing(), "dim:", fixed.GetDimension())
    print("Moving size:", moving.GetSize(), "spacing:", moving.GetSpacing(), "dim:", moving.GetDimension())

    # -----------------------------
    # Load parameter files
    # Put them in the order you want elastix to apply them
    # -----------------------------
    param_dir = Path(__file__).parent / "ParameterFiles"
    rigid_p  = param_dir / "Rigid1.txt"
    affine_p = param_dir / "Affine1.txt"
    bspl_p   = param_dir / "BSpline1.txt"

    for p in (rigid_p, affine_p, bspl_p):
        if not p.exists():
            raise FileNotFoundError(f"Parameter file not found: {p}")

    pms = []
    for p in (rigid_p, affine_p, bspl_p):
        pm = sitk.ReadParameterFile(str(p))
        pms.append(pm)

        # Print the most common crash-related parameters
        img_dim = _pm_get(pm, "ImageDimension")
        nor = _pm_get(pm, "NumberOfResolutions")
        gss = _pm_get(pm, "GridSpacingSchedule")

        print("\nPARAM FILE:", p.resolve())
        print("  ImageDimension:", img_dim)
        print("  NumberOfResolutions:", nor)
        print("  GridSpacingSchedule:", gss)

        # Detect the classic elastix mismatch before executing
        if img_dim and nor and gss:
            try:
                dim_i = int(img_dim[0])
                nor_i = int(nor[0])
                gss_len = len(gss)
                exp1 = nor_i
                exp2 = nor_i * dim_i
                print(f"  GridSpacingSchedule length = {gss_len} (expected {exp1} or {exp2})")
                if gss_len not in (exp1, exp2):
                    print("  >>> MISMATCH DETECTED: elastix will likely crash with Invalid GridSpacingSchedule.")
            except Exception as e:
                print("  Could not parse schedule lengths:", e)

    # -----------------------------
    # Setup elastix
    # -----------------------------
    elx = sitk.ElastixImageFilter()
    elx.SetFixedImage(fixed)
    elx.SetMovingImage(moving)
    elx.SetOutputDirectory(str(out_dir))
    elx.SetParameterMap(pms)

    # MUST: enable logs so the internal elastix error is visible
    elx.LogToConsoleOn()
    elx.LogToFileOn()
    elx.SetLogFileName("elastix.log")

    # -----------------------------
    # Execute
    # -----------------------------
    print("\nStarting elastix...")
    try:
        elx.Execute()
    except Exception:
        print("\nElastix crashed. Open the log for the real reason:")
        print(str(out_dir / "elastix.log"))
        raise

    # -----------------------------
    # Save result
    # -----------------------------
    result = elx.GetResultImage()
    out_img = out_dir / "result.nii.gz"
    sitk.WriteImage(result, str(out_img))
    print("\nDONE. Result saved to:", out_img)


if __name__ == "__main__":
    # CHANGE ONLY THESE 3 PATHS
    FIXED = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train\040\t2.nii.gz")
    MOVING = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita\prostate158_train\train\020\t2.nii.gz")
    OUT = Path(r"C:\Users\30697\OneDrive\2.Netherlands\capita\atlas_results\debug_020_to_040")

    run_registration(FIXED, MOVING, OUT)