from prostateLoader import ProstateLoader
from utils import final_metric_from_elastix_log

import SimpleITK as sitk
import time
import matplotlib.pyplot as plt

# Timer start of program
t0 = time.perf_counter()

# Variables
ATLAS_SIZE = 50
PRESELECTION_SIZE = 5
VISUALISATION_SLICE = 15

# Load data
loader = ProstateLoader()
images, segmen = loader.LoadData()

# Divide into atlas and test images
atlas_images = images[0:ATLAS_SIZE]
atlas_segmen = segmen[0:ATLAS_SIZE]

test_images = images[ATLAS_SIZE:]
test_segmen = segmen[ATLAS_SIZE:]

# Timer for loading
t1 = time.perf_counter()

# Select an example test image
fixed_img = test_images[1]

# Elastix registration object
elx = sitk.ElastixImageFilter()
elx.SetFixedImage(fixed_img)

# Load parameter maps from file
pm_affine = sitk.ReadParameterFile("ParameterFiles/Affine/affine.txt")

# Set registration parameters
elx.SetParameterMap(pm_affine)
elx.LogToConsoleOff()
elx.LogToFileOn()

metrics = []
transforms = []
reg_results = []
reg_segmen = []

for idx, moving_img in enumerate(atlas_images):

    # Run
    elx.SetMovingImage(moving_img)
    elx.Execute()

    registered_img = elx.GetResultImage()
    tmap = elx.GetTransformParameterMap()  # save this to apply to masks or other images

    fixed_np = sitk.GetArrayFromImage(fixed_img)  # (Z,Y,X)
    reg_np = sitk.GetArrayFromImage(registered_img)  # (Z,Y,X)

    metric = final_metric_from_elastix_log()
    metrics.append(metric)
    reg_results.append(registered_img)
    transforms.append(tmap)

# Timer for end of affine registrations
t2 = time.perf_counter()

# Zip all the data togehter
results = list(zip(metrics, transforms, atlas_images, atlas_segmen, reg_results))

# Sort by metric, highest first
results_sorted = sorted(results, key=lambda t: t[0], reverse=True)

# Take top x
top_results = results_sorted[:PRESELECTION_SIZE]

# Unpack again
top_metrics = [r[0] for r in top_results]
top_tmaps = [r[1] for r in top_results]
top_images = [r[2] for r in top_results]
top_segmen = [r[3] for r in top_results]
top_reg = [r[4] for r in top_results]

t3 = time.perf_counter()

plt.figure()

plt.subplot(2, 3, 1)
plt.imshow(fixed_img[:, :, VISUALISATION_SLICE], cmap="gray")
plt.title("Test image")

for i, img in enumerate(top_reg):
    plt.subplot(2, 3, i + 2)
    plt.imshow(img[:, :, VISUALISATION_SLICE], cmap="gray")
    plt.title(f"NCC: {top_metrics[i]:.3f}")

t4 = time.perf_counter()

print("___________________________________________")
print(f"Loading the images      : {(t1 - t0)//60:.0f}m {(t1 - t0)%60:.2f}s")
print(f"Affine registrations    : {(t2 - t1)//60:.0f}m {(t2 - t1)%60:.2f}s")
print(f"Preselection            : {(t3 - t2)//60:.0f}m {(t3 - t2)%60:.2f}s")
print(f"Plotting                : {(t4 - t3)//60:.0f}m {(t4 - t3)%60:.2f}s")
print("_____________________________________")
print(f"Total time              : {(t4 - t0)//60:.0f}m {(t4 - t0)%60:.2f}s")

plt.show()
