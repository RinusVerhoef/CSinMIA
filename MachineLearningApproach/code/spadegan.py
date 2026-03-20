import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

l1_loss = torch.nn.L1Loss()


class Block(nn.Module):
    """Basic convolutional building block

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)  # batch normalisation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)  # batch normalisation
        self.lrelu = nn.LeakyReLU(0.05)  # leaky ReLU

    def forward(self, x):
        """Performs a forward pass of the block

        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        return x


class Encoder(nn.Module):
    """The encoder part of the VAE.

    Parameters
    ----------
    spatial_size : list[int]
        size of the input image, by default [64, 64]
    z_dim : int
        dimension of the latent space
    chs : tuple
        hold the number of input channels for each encoder block
    """

    def __init__(self, spatial_size=[64, 64], z_dim=256, chs=(1, 64, 128, 256)):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # height and width of images at lowest resolution level
        _h, _w = spatial_size[0] // (2 ** (len(chs) - 1)), spatial_size[1] // (
            2 ** (len(chs) - 1)
        )

        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input image to the encoder

        Returns
        -------
        list[torch.Tensor]
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """

        for block in self.enc_blocks:
            x = block(x)  # conv block
            x = self.pool(x)  # pooling
        x = self.out(x)  # output layer
        return torch.chunk(x, 2, dim=1)  # 2 chunks, 1 each for mu and logvar


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = segmap.float()
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = in_ch != out_ch
        mid_ch = min(in_ch, out_ch)

        # create conv layers
        self.conv_0 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.norm_0 = SPADE(in_ch, semantic_nc)
        self.norm_1 = SPADE(mid_ch, semantic_nc)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_ch, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class SPADE_Generator(nn.Module):
    """Generator of the VAE

    Parameters
    ----------
    z_dim : int
        dimension of latent space
    chs : tuple
        holds the number of channels for each block
    h : int, optional
        height of image at lowest resolution level, by default 8
    w : int, optional
        width of image at lowest resolution level, by default 8
    """

    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), semantic_nc=1, h=8, w=8):

        super().__init__()
        self.chs = chs
        self.h = h
        self.w = w
        self.z_dim = z_dim
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping

        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=2, stride=2)
                for i in range(len(chs) - 1)
            ]
        )  # transposed convolution

        self.dec_blocks = nn.ModuleList(
            [
                SPADEResnetBlock(chs[i + 1], chs[i + 1], semantic_nc)
                for i in range(len(chs) - 1)
            ]
        )  # SPADE block
        self.head = nn.Sequential(
            nn.Conv2d(self.chs[-1], 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )  # output layer

    def forward(self, z, seg):
        """Performs the forward pass of generator

        Parameters
        ----------
        z : torch.Tensor
            input to the generator

        Returns
        -------
        x : torch.Tensor

        """
        x = self.proj_z(z)  # fully connected layer
        x = self.reshape(x)  # reshape to image dimensions
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)  # transposed convolution
            x = self.dec_blocks[i](x, seg)  # SPADE block
        return self.head(x)


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Flatten(1, -1),
            # This should change dynamicly
            nn.Linear(8192, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class SPADE_GAN(nn.Module):
    """A representation of the VAE

    Parameters
    ----------
    enc_chs : tuple
        holds the number of input channels of each block in the encoder
    dec_chs : tuple
        holds the number of input channels of each block in the decoder
    """

    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
    ):
        super().__init__()
        self.encoder = Encoder(chs=enc_chs)
        self.generator = SPADE_Generator(chs=dec_chs)
        self.discriminator = Discriminator(ndf=64)

    def forward(self, x, seg):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.

        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder

        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)

        output = self.generator(latent_z, seg)

        return output, mu, logvar

    def discriminate(self, x):
        return self.discriminator(x)


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.

    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with
    random numbers from the normal distribution.

    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted
    latent distribution.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution

    Returns
    -------
    float
        the kld loss

    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(inputs, recons, mu, logvar, beta=1):
    """Computes the VAE loss, sum of reconstruction and KLD loss

    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution

    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    return l1_loss(inputs, recons) + beta * kld_loss(mu, logvar)
