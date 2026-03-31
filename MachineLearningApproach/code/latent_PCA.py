import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from u_net import UNet
from utils import ProstateMRDataset

# Parameters and paths
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 16
DATA_DIR = Path("prostate158_train/train")
WEIGHTS = Path("segmentation_model_weights/u_net.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained U-Net model
model = UNet().to(device)
state_dict = torch.load(WEIGHTS, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Load dataset
patients = sorted([
    p for p in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in p.parts)
])

dataset = ProstateMRDataset(patients, IMAGE_SIZE, valid=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Extract latent representations for all slices, keeping track of patient IDs
latents = []
patient_ids = []

with torch.no_grad():
    for idx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)

        _, latent = model(imgs, return_latent=True)

        latent = latent.view(latent.size(0), -1).cpu().numpy()
        latents.append(latent)

        # Track which patient each slice belongs to
        start = idx * BATCH_SIZE
        end   = start + latent.shape[0]

        # Dataset has fixed 24 slices per patient
        for i in range(latent.shape[0]):
            patient_ids.append(start // 24)

latents = np.concatenate(latents, axis=0)

# PCA
pca = PCA(n_components=2)
xy = pca.fit_transform(latents)

plt.figure(figsize=(10,10))
scatter = plt.scatter(xy[:,0], xy[:,1], c=patient_ids, cmap='tab20', s=10)
plt.colorbar(scatter, label="Patient ID")
plt.title("PCA of U-Net Latent Space (per slice)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("all_slices_pca.png", dpi=300)
plt.show()