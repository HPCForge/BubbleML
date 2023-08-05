from typing import List, Tuple, Union

import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    """2D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes to keep in the first spatial direction
        modes2 (int): Number of Fourier modes to keep in the second spatial direction
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )

    def forward(self, x, x_dim=None, y_dim=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = SpectralConv2d._batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], torch.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = SpectralConv2d._batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], torch.view_as_complex(self.weights2)
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
    @staticmethod
    def _batchmul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    

class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class FourierResidualBlock(nn.Module):
    """Fourier Residual Block to be used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of modes in the first dimension.
        modes2 (int): Number of modes in the second dimension..
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.fourier1 = SpectralConv2d(in_channels, out_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros")
        self.fourier2 = SpectralConv2d(out_channels, out_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros")
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        # using pre-norms
        h = self.activation(self.norm1(x))
        x1 = self.fourier1(h)
        x2 = self.conv1(h)
        out = x1 + x2
        out = self.activation(self.norm2(out))
        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2 + self.shortcut(x)
        return out


class DownBlock(nn.Module):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor):
        return self.res(x)


class FourierDownBlock(nn.Module):
    """Down block This combines [`FourierResidualBlock`][pdearena.modules.twod_unet.FourierResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        norm: bool = False,
    ):
        super().__init__()
        self.res = FourierResidualBlock(
            in_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        return self.res(x)


class UpBlock(nn.Module):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor):
        return self.res(x)


class FourierUpBlock(nn.Module):
    """Up block that combines [`FourierResidualBlock`][pdearena.modules.twod_unet.FourierResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Note:
        We currently don't recommend using this block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = FourierResidualBlock(
            in_channels + out_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        return self.res(x)


class MiddleBlock(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, n_channels: int, norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, norm=norm)
        self.res2 = ResidualBlock(n_channels, n_channels, norm=norm)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)



class FourierUnet(nn.Module):
    """Unet with Fourier layers in early downsampling blocks.

    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        hidden_channels (int): Number of channels in the first layer.
        modes1 (int): Number of Fourier modes to use in the first spatial dimension.
        modes2 (int): Number of Fourier modes to use in the second spatial dimension.
        norm (bool): Whether to use normalization.
        n_blocks (int): Number of blocks to use at each resolution.
        n_fourier_layers (int): Number of early downsampling layers to use Fourier layers in.
        mode_scaling (bool): Whether to scale the number of modes with resolution.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        n_blocks: int = 2,
        n_fourier_layers: int = 2,
        mode_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = self.input_channels
        n_channels = hidden_channels
        # Project image into feature map
        self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            if i < n_fourier_layers:
                for _ in range(n_blocks):
                    down.append(
                        FourierDownBlock(
                            in_channels,
                            out_channels,
                            modes1=max(modes1 // 2**i, 4) if mode_scaling else modes1,
                            modes2=max(modes2 // 2**i, 4) if mode_scaling else modes2,
                            norm=norm,
                        )
                    )
                    in_channels = out_channels
            else:
                # Add `n_blocks`
                for _ in range(n_blocks):
                    down.append(
                        DownBlock(
                            in_channels,
                            out_channels,
                            norm=norm,
                        )
                    )
                    in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.output_channels

        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor):
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        x = self.final(self.activation(self.norm(x)))
        return x