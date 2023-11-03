import torch.nn.functional as F
import torch
import torch.nn as nn
from .gfno import GConv2d, GMLP2d, GNorm
from functools import partial
# ----------------------------------------------------------------------------------------------------------------------
# GCNN2d
# ----------------------------------------------------------------------------------------------------------------------
class GCNN2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, reflection):
        super(GCNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.kernel_size = 3
        assert self.kernel_size % 2 == 1, "Kernel size should be odd"
        self.padding = (self.kernel_size - 1) // 2
        self.pad = partial(torch.nn.functional.pad, pad=[self.padding] * 4)#, mode="circular")

        self.width = width

        self.p = GConv2d(in_channels=in_channels, out_channels=self.width, kernel_size=1,
                         reflection=reflection, first_layer=True)  # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
        self.conv1 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
        self.conv2 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
        self.conv3 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=self.kernel_size, reflection=reflection)
        self.mlp0 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp1 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp2 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.mlp3 = GMLP2d(in_channels=self.width, out_channels=self.width, mid_channels=self.width, reflection=reflection)
        self.w0 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.w1 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.w2 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.w3 = GConv2d(in_channels=self.width, out_channels=self.width, kernel_size=1, reflection=reflection)
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))
        self.q = GMLP2d(in_channels=self.width, out_channels=out_channels, mid_channels=self.width * 4, reflection=reflection,
                        last_layer=True)  # output channel is 1: u(x, y)

    def forward(self, x):
        x = self.p(x)

        x1 = self.norm(self.conv0(self.pad(self.norm(x))))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.pad(self.norm(x))))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.pad(self.norm(x))))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.pad(self.norm(x))))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.q(x)
        return x#.unsqueeze(-2)
