import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from engine.extre_module.ultralytics_nn.conv import Conv, DWConv
from engine.extre_module.ultralytics_nn.block import Bottleneck


class MI_Block(nn.Module):
    """
    Multi-scale Interaction Block that combines multi-branch features.
    """

    def __init__(self, c1, c2, module=partial(Bottleneck, shortcut=False, k=((3, 3), (3, 3)), e=1.0, g=1),
                 n=1, p=1, kernel_size=3, e=0.5):
        super().__init__()
        # Compute intermediate channel dimension
        self.c = int(c2 * e)

        # Initial convolution: expand channels
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)

        # Final convolution: project to output channels
        self.cv_final = Conv((4 + n) * self.c, c2, 1)

        # Module list for repeated blocks (e.g., Bottleneck)
        self.m = nn.ModuleList(module(self.c, self.c) for _ in range(n))

        # Branch 1: simple 1x1 conv
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)

        # Branch 2: 1x1 conv -> depthwise conv -> 1x1 conv
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            DWConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1)
        )

    def forward(self, x):
        # Initial feature expansion
        y = self.cv_first(x)

        # Generate four base branches
        y0 = self.cv_block_1(y)          # branch 0: 1x1 conv
        y1 = self.cv_block_2(y)          # branch 1: depthwise conv path
        y2, y3 = y.chunk(2, dim=1)       # branch 2 & 3: split the expanded features

        # Collect all branches
        branches = [y0, y1, y2, y3]

        # Append outputs of repeated modules, each taking the last branch as input
        branches.extend(m(branches[-1]) for m in self.m)

        # Concatenate all branches and project to target channels
        return self.cv_final(torch.cat(branches, dim=1))


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # Instantiate MI_Block with two repeated modules
    module = MI_Block(in_channel, out_channel, n=2).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(
        model=module,
        input_shape=(batch_size, in_channel, height, width),
        output_as_string=True,
        output_precision=4,
        print_detailed=True
    )
    print(RESET)