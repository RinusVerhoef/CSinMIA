import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
import spadegan

# to ensure reproducible training/validation split
random.seed(42)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

DATA_DIR = Path.cwd() / "prostate158_train" / "train"
CHECKPOINTS_DIR = Path.cwd() / "SPADE_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "SPADE_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 20
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 64
N_EPOCHS = 200
DECAY_LR_AFTER = 100
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10

# dimension of VAE latent space
Z_DIM = 128

LAMBDA_REC = 10.0
LAMBDA_KL = 3.0
LAMBDA_GAN = 3.0


# function to reduce the
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )


# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling

dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE, valid=True)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser
model = spadegan.SPADE_GAN().to(device)

optimizer_G = torch.optim.Adam(
    list(model.encoder.parameters()) + list(model.generator.parameters()),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)

optimizer_D = torch.optim.Adam(
    model.discriminator.parameters(),
    lr=LEARNING_RATE * 0.8,
    betas=(0.5, 0.999),
)

# add a learning rate scheduler based on the lr_lambda function
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)


bce_loss = nn.BCELoss()

best_valid_loss = float("inf")

# training loop
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch+1}/{N_EPOCHS}")
    model.train()  # set model to training mode

    current_train_rec = 0.0
    current_train_kl = 0.0
    current_train_gan_g = 0.0
    current_train_gan_d = 0.0
    current_train_total_g = 0.0

    # training iterations
    for x_real, y_real in tqdm(dataloader, position=0):
        x_real = x_real.to(device)
        y_real = y_real.to(device)

        # Discriminator
        optimizer_D.zero_grad()  # zero the gradients
        with torch.no_grad():
            x_recon, mu, logvar = model(x_real, y_real)  # forward pass

        pred_real = model.discriminate(x_real)
        pred_fake = model.discriminate(x_recon.detach())

        real_labels = torch.ones_like(pred_real)
        fake_labels = torch.zeros_like(pred_fake)

        loss_d_real = bce_loss(pred_real, real_labels)
        loss_d_fake = bce_loss(pred_fake, fake_labels)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        loss_d.backward()
        optimizer_D.step()

        # Encoder and SPADE Generator
        optimizer_G.zero_grad()

        x_recon, mu, logvar = model(x_real, y_real)  # forward pass
        pred_fake_for_g = model.discriminate(x_recon)

        # loss = spadegan.vae_loss(x_real, x_recon, mu, logvar)  # compute loss
        loss_rec = nn.functional.l1_loss(x_recon, x_real)
        loss_kl = spadegan.kld_loss(mu, logvar)
        loss_g_gan = bce_loss(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

        loss_g = LAMBDA_REC * loss_rec + LAMBDA_KL * loss_kl + LAMBDA_GAN * loss_g_gan
        loss_g.backward()  # backpropagate
        optimizer_G.step()  # update weights

        current_train_rec += loss_rec.item()
        current_train_kl += loss_kl.item()
        current_train_gan_g += loss_g_gan.item()
        current_train_gan_d += loss_d.item()
        current_train_total_g += loss_g.item()

    writer.add_scalar("Loss/train_recon", current_train_rec / len(dataloader), epoch)
    writer.add_scalar("Loss/train_kl", current_train_kl / len(dataloader), epoch)
    writer.add_scalar("Loss/train_gan_g", current_train_gan_g / len(dataloader), epoch)
    writer.add_scalar("Loss/train_gan_d", current_train_gan_d / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/train_total_g", current_train_total_g / len(dataloader), epoch
    )

    scheduler_G.step()
    scheduler_D.step()

    # evaluate validation loss
    model.eval()
    current_valid_rec = 0.0
    current_valid_kl = 0.0
    current_valid_total = 0.0

    with torch.no_grad():
        for x_real, y_real in tqdm(valid_dataloader, position=0):
            x_real = x_real.to(device)
            y_real = y_real.to(device)

            x_recon, mu, logvar = model(x_real, y_real)  # forward pass

            loss_rec = nn.functional.l1_loss(x_recon, x_real)
            loss_kl = spadegan.kld_loss(mu, logvar)
            loss_total = LAMBDA_REC * loss_rec + LAMBDA_KL * loss_kl

            current_valid_rec += loss_rec.item()
            current_valid_kl += loss_kl.item()
            current_valid_total += loss_total.item()

        # write to tensorboard log
        writer.add_scalar(
            "Loss/valid_recon", current_valid_rec / len(valid_dataloader), epoch
        )
        writer.add_scalar(
            "Loss/valid_kl", current_valid_kl / len(valid_dataloader), epoch
        )
        writer.add_scalar(
            "Loss/valid_total", current_valid_total / len(valid_dataloader), epoch
        )

        avg_valid_loss = current_valid_rec / len(valid_dataloader)
        writer.add_scalar("Loss/validation", avg_valid_loss, epoch)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            weights_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(weights_dict, CHECKPOINTS_DIR / "SpadeBestModel.pth")
            print(
                f"Saved new best model at epoch {epoch + 1} "
                f"with val loss {avg_valid_loss:.4f}"
            )

        # save examples of real/fake images
        if (epoch + 1) % DISPLAY_FREQ == 0:
            img_grid = make_grid(
                torch.cat((x_recon[:5].cpu(), x_real[:5].cpu())), nrow=5, padding=12, pad_value=-1
            )
            writer.add_image(
                "Real_fake",
                np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5,
                epoch + 1,
            )

            z = spadegan.get_noise(10, Z_DIM, device=device)  # sample noise
            # Generate 10 images and display
            seg_sample = y_real[:10]
            image_samples = model.generator(z, seg_sample)  # generate 10 images
            img_grid = make_grid(
                torch.cat((image_samples[:5].cpu(), image_samples[5:].cpu())),
                nrow=5,
                padding=12,
                pad_value=-1,
            )
            writer.add_image(
                "Samples",
                np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5,
                epoch + 1,
            )


weights_dict = {k: v.cpu() for k, v in model.state_dict().items()}
torch.save(
    weights_dict,
    CHECKPOINTS_DIR / "SpadeFinalModel.pth",
)
writer.close()
