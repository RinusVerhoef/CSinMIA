from prostateLoader import ProstateLoader

import matplotlib.pyplot as plt

loader = ProstateLoader()

images, segment = loader.LoadData()


idx = 50
im = images[idx]
seg = segment[idx]


# Pick a slice you want to visualize
slice_idx = im.shape[2] // 2  # middle slice

# Plot image with segmentation overlay
plt.figure()
plt.imshow(im[:, :, slice_idx].T, cmap="gray")
plt.imshow(seg[:, :, slice_idx].T, cmap="jet", alpha=0.3)
plt.axis("off")
plt.show()
