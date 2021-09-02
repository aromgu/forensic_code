import torch
import torch.nn as nn

from .Unet import Unet as unet
from .cnnwav import CNN
from .MGP import extract as mgp
from .MGP import lowfreq_mask as lowfreq
from .Canny import canny
from utils.get_functions import get_init
args = get_init()

class Ours(nn.Module):
    def __init__(self):
        super(Ours, self).__init__()

        # self.Unet = unet
        self.CNN = CNN()
        self.MGP = mgp
        self.LowFreq = lowfreq
        self.Canny = canny


    def forward(self, X, y, device):
        mgps, spectrum, mask_list = self.MGP(X, device, args.img_size, args.angle, args.length, args.preserve_range, args.num_enc)
        mgps = mgps.unsqueeze(dim=2)
        k, b, c, w, h = mgps.shape  # [num_enc,B,C,256,256]

        pred_list = torch.empty(k, b, c, w, h)
        for k_ in range(k):
            prediction = self.CNN(mgps[k_])
            pred_list[k_] = prediction
        mgp_output = torch.mean(pred_list, dim=0)

        # EDGE GT
        edge_GT = self.Canny(y, device)
        # LOW FREQ INPUT
        lowfreq = self.LowFreq(15, X)

        return mgp_output, edge_GT, lowfreq, spectrum, mask_list, k