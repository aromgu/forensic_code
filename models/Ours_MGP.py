import torch
import torch.nn as nn
import torch.fft as fft

import numpy as np
import kornia.color as k_color

# from .Unet import Unet as unet
from .PretrainedResnet import ResCNN
# from .RRUNet import Ringed_Res_Unet
# from .resnet import resnet18
# from .Canny import canny

from .MGP import get_small_region, fourier_intensity_extraction

class Ours(nn.Module):
    def __init__(self, args, device):
        super(Ours, self).__init__()

        self.device = device
        self.preserve_range = args.preserve_range
        self.num_enc = args.num_enc
        self.img_size = args.img_size

        self.idxx, self.idxy = get_small_region(args.img_size, args.angle, args.length, args.preserve_range)

        # self.Unet = unet
        self.cnn_low_freq = ResCNN()
        self.cnn_high_freq = ResCNN()
        # self.Canny = canny


    def forward(self, image, gt, device):
        Y, U, V = self._RGB2YUV(image) # apply transform only Y channel

        # low frequency filtering
        low_filtered_image = self.low_frequency_filtering(Y).repeat(1, 3, 1, 1)
        low_filtered_image[..., 1, :, :] = U
        low_filtered_image[..., 2, :, :] = V
        low_filtered_image = self._YUV2RGB(low_filtered_image) # convert to YUV -> RGB
        low_freq_output = self.cnn_low_freq(low_filtered_image)

        # Get high frequency
        patterns = self.extract_pattern(Y)
        pred_list = torch.empty_like(patterns)
        for k_ in range(pred_list.size(0)):
            prediction = self.cnn_high_freq(patterns[k_])
            pred_list[k_] = prediction
        high_freq_output = torch.mean(pred_list, dim=0)

        # EDGE GT
        # edge_GT = self.Canny(gt, device)

        return high_freq_output, low_freq_output#, edge_GT

    def low_frequency_filtering(self, image):
        # low frequency mask generation
        mask = torch.zeros((image.size(2), image.size(3)), dtype=image.dtype).to(self.device)

        x_range = np.arange(0, image.size(2)) - int(image.size(2) / 2)
        y_range = np.arange(0, image.size(3)) - int(image.size(3) / 2)
        x_ms, y_ms = np.meshgrid(x_range, y_range)
        R = np.sqrt(x_ms ** 2 + y_ms ** 2)

        idxx, idxy = np.where(R <= self.preserve_range)

        mask[idxx, idxy] = 1

        # frequency filtering
        image_fft = fft.fftshift(fft.fft2(image))
        image_fft *= mask
        low_filtered_image = torch.abs(fft.ifft2(fft.ifftshift(image_fft)))

        return low_filtered_image

    def extract_pattern(self, image):
        image_fft = fft.fftshift(fft.fft2(image))
        spectrum = torch.log(1 + torch.abs(image_fft))
        self.clustered_idx, self.X, self.labels = fourier_intensity_extraction(spectrum, self.idxx, self.idxy, self.num_enc, self.img_size)
        patterns = torch.empty((self.num_enc, image_fft.size(0), image_fft.size(1), image_fft.size(2), image_fft.size(3))).to(self.device)

        for i, (idxx, idxy) in enumerate(self.clustered_idx):
            mask = torch.zeros((image_fft.size(2), image_fft.size(3))).to(self.device)
            mask[idxx, idxy] = 1
            temp = torch.empty_like(image_fft, dtype=torch.float)
            for j in range(len(image_fft)):
                temp[j] = torch.abs(fft.ifft2(fft.ifftshift(image_fft[j] * mask)))
            patterns[i] = temp

        return patterns

    def _RGB2YUV(self, img):
        YUV = k_color.rgb_to_yuv(img)

        return YUV[..., 0, :, :].unsqueeze(dim=1), YUV[..., 1, :, :], YUV[..., 2, :, :]

    def _YUV2RGB(self, img):
        RGB = k_color.yuv_to_rgb(img)

        return RGB

