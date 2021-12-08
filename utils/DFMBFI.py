import torch
import torch.nn as nn

class AFIMB(nn.Module):
    def __init__(self, ch, w):
        super(AFIMB, self).__init__()

        self.w = w
        self.conv1 = nn.Conv2d(ch, ch//2, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d((ch//2)*3, ch//2, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0))#, dilation=(1,1))
        self.linear1 = nn.Linear((ch//2)*(w//2)*(w//2), ch//2)
        self.linear2 = nn.Linear(ch//2,ch//2)
        self.convone = nn.Conv2d(ch//2, ch, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.channel_att = eca(ch, w)

        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, rgb, low, high):
        rgb = self.conv1(rgb)
        low = self.conv1(low)
        high = self.conv1(high)

        cat = torch.cat((rgb, low, high), dim=1) # 192

        x = self.conv2(cat) # 192
        pool = self.maxpool(x)

        linear1 = self.linear1(torch.flatten(pool, start_dim=1))
        linear2 = self.linear2(linear1)

        linear2 = linear2.unsqueeze(2).unsqueeze(3)
        linear2 = linear2.repeat(1,1,self.w,self.w)
        pool = self.upscale(pool)

        element_mul = linear2 * pool

        cha_att = self.channel_att(element_mul)
        convone = self.convone(cha_att)

        return convone

class eca(nn.Module):
    def __init__(self, ch, w):
        super(eca, self).__init__()
        self.ch = ch
        self.w = w
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1, kernel_size=(3), padding=(1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = y.repeat(1,self.ch//2,8,self.w)

        return x * y[:,:,:self.w,:]

if __name__=='__main__':
    inp = torch.rand((2, 512, 128, 128)).cuda()
    model = AFIMB(512, 128).cuda()
    out = model(inp, inp, inp)
