import torch
import torch.nn as nn
import torch.nn.functional as F

class GATE(nn.Module):
    def __init__(self, h, c):
        super(GATE, self).__init__()
        self.flat = nn.AvgPool2d(kernel_size=(h, h))
        self.lin = nn.Linear(c*3, 3)
        self.c = c
        self.h = h

    def forward(self, rgb, low, high):
        # x = [b, c, h, w]
        flat_rgb = self.flat(rgb).squeeze(-1).squeeze(-1) # [b, c]
        flat_low = self.flat(low).squeeze(-1).squeeze(-1) # [b, c]
        flat_high = self.flat(high).squeeze(-1).squeeze(-1) # [b, c]

        cat = torch.cat([flat_rgb, flat_low, flat_high], dim=1) # [b, c*3]
        lin = self.lin(cat) # [b, 3]
        lin =F.softmax(lin, dim=-1) # att

        lin_0 = lin[:,0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1,self.c,self.h,self.h)
        lin_1 = lin[:,1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1,self.c,self.h,self.h)
        lin_2 = lin[:,2].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1,self.c,self.h,self.h)

        flat_rgb_ = rgb*lin_0
        flat_low_ = low*lin_1
        flat_high_ = high*lin_2

        out = flat_rgb_ + flat_low_ + flat_high_ # 인풋이랑 같음 ㅋㅋㄹㅃㅃ
        # out = torch.cat([flat_rgb_, flat_low_, flat_high_], dim=1)

        return out
