import torch
import torch.nn as nn
import torch.fft as fft

import numpy as np
import kornia.color as k_color
# from .PretrainedASPP import ResASPP
from .PretrainedResnet import ResCNN

class Patchwork(nn.Module):
    def __init__(self, device):
        super(Patchwork, self).__init__()

        self.device = device
        self.rescnn = ResCNN()

    def forward(self, image):
        patches = self.extract_patch(image)
        predict = self.rescnn(patches)
        return predict

    def extract_patch(self, image, channel):
        image = image.unsqueeze(dim=0)
        patches = self.unfold(image)
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape(-1, channel, self.patch_size, self.patch_size) # [P*B, C, W, H]

        return patches
