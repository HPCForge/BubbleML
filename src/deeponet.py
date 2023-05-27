"""
U-Net. Implementation taken and modified from
https://github.com/mateuszbuda/brain-segmentation-pytorch

MIT License

Copyright (c) 2019 mateuszbuda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
from collections import OrderedDict
import torch
from torch import nn
from .unet import UNet2d

class DeepONet(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.branch = self._build_branch()
        self.trunk = self._build_trunk() 

    def _build_branch(self):
        return = nn.Sequential(
            UNet2d(self.in_channels, self.out_channels),
            nn.Flatten(),
            nn.Linear(512*512, 512),
            nn.Linear(512, 512)
            nn.Linear(512, 128)
        )

    def _build_trunk(self):
        return nn.Sequential(
            nn.Linear(2, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128)
        )

    def forward(self, input, xy):
        r""" for each input, predict all xy
          input: mxd
          xy:    nx2
          output: mxnx1
        """
        # [M x D] -> [M x 128]
        b = self.branch(input)
        # [N x 2] -> [N x 128]
        t = self.trunk(xy)
        # MxNx1
        return (b @ t.transpose()).unsqueeze(-1)
