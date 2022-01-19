import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class _ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()

        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0, dilation=1, bias=False)
        self.conv3x3_1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1),  padding=rates[0], dilation=rates[0], bias=False)
        self.conv3x3_2 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1),  padding=rates[1], dilation=rates[1], bias=False)
        self.conv3x3_3 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=rates[2], dilation=rates[2], bias=False)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)

        tensor = torch.cat([x1, x2, x3, x4], dim=1)

        return tensor

        # tensor = torch.cat([stage(x) for stage in self.children()], dim=1)
        # return tensor


if __name__ == '__main__':
    import cv2
    src = cv2.imread('./img1.tif')
    src = cv2.resize(src,(256,256))
    src = torch.from_numpy(src).permute(2, 1, 0).unsqueeze(dim=0).float()
    model = _ASPP(3,3,[6,12,18,24])
    dst = model(src)
    plt.imshow(dst[0].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
    plt.show()