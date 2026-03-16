import torch
import torch.nn as nn


class Block(nn.Module):
    """A representation for the basic convolutional building block of the unet

    Parameters
    ----------
    in_ch : int 
        number of input channels to the block
    out_ch : int 
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Performs the forward pass of the block.

        Parameters
        ----------
        x : torch.Tensor
            the input to the block

        Returns
        -------
        x : torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """A representation for the encoder part of the unet.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the encoder

    """

    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input 

        Returns
        -------
        list[torch.Tensor]
            contains the outputs of each block in the encoder
        """
        ftrs = []  # a list to store features
        for block in self.enc_blocks[:-1]:  # for all blocks except the last one
            x = block(x)        
            # # save features to concatenate to decoder blocks
            ftrs.append(x)          # save features for skip connections
            x = self.pool(x)
        # for the last block, we don't apply max pooling
        x = self.enc_blocks[-1](x)
        ftrs.append(x) # bottom feature
        return ftrs


class Decoder(nn.Module):
    """A representation for the decoder part of the unet.

    Layers consist of transposed convolutions followed by convolutional blocks.

    Parameters
    ----------
    chs : tuple
        holds the number of input channels of each block in the decoder
    """

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=2, stride=2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(2 * chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        """Performs the forward pass for all blocks in the decoder. 
        
        Parameters
        ----------
        x : torch.Tensor
            input to the decoder
        encoder_features: list
            contains the encoder features to be concatenated to the corresponding level of the decoder

        Returns
        -------
        x : torch.Tensor
            output of the decoder 
        """
        for i in range(len(self.chs) - 1):
            # transposed convolution
            x = self.upconvs[i](x)
            # get the features from the corresponding level of the encoder
            enc_f = encoder_features[-(i + 2)]
            # concatenate these features to x
            x = torch.cat([x, enc_f], dim=1)
            # convolutional block
            x = self.dec_blocks[i](x)


        return x


class UNet(nn.Module):
    """A representation for a unet
    
    Parameters
    ----------
    enc_chs : tuple
        holds the number of input channels of each block in the encoder
    dec_chs : tuple
        holds the number of input channels of each block in the decoder
    num_classes : int
        number of output classes of the segmentation
    """

    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),          # right now, 2 times pooling is applied, and 2 times upsampling
        dec_chs=(256, 128, 64),             # can be changed to apply more pooling/upsampling
        num_classes=1,
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], num_classes, 1),
        )  # output layer
        assert enc_chs[-1] == dec_chs[0], (
            f"Decoder expects bottom channels {dec_chs[0]} "
            f"to match encoder's last output {enc_chs[-1]}"
        )
        assert len(dec_chs) == len(enc_chs) - 1, (
            "Number of decoder stages must equal number of encoder poolings: "
            f"got len(dec_chs)={len(dec_chs)} vs len(enc_chs)-1={len(enc_chs)-1}"
        )
    def forward(self, x):
        """Performs the forward pass of the unet.
       
        Parameters
        ----------
        x : torch.Tensor
            the input to the unet (image)

        Returns
        -------
        out : torch.Tensor
            unet output, the logits of the predicted segmentation mask
        """
       
        enc_ftrs = self.encoder(x)
        x = self.decoder(enc_ftrs[-1], enc_ftrs)
        out = self.head(x)
        return out

