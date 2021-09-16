import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        tensor = torch.cat([stage(x) for stage in self.children()], dim=1)
        return tensor


if __name__ == '__main__':
    import cv2
    src = cv2.imread('./img1.tif')
    src = cv2.resize(src,(256,256))
    src = torch.from_numpy(src).permute(2, 1, 0).unsqueeze(dim=0).float()
    model = _ASPP(3,3,[6,12,18,24])
    dst = model(src)
    plt.imshow(dst[0].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
    plt.show()