import torch
import torch.nn as nn

class RFAM(nn.Module):
    def __init__(self, ch):
        super(RFAM, self).__init__()
        self.conv1x1 = nn.Conv2d(ch*3, ch, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        # self.conv1x1 = nn.Conv2d(ch, ch, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def forward(self, rgb, low, high):
        cat = torch.cat((rgb, low, high),dim=1)

        convone = self.conv1x1(cat)
        bn = self.bn(convone)
        relu = self.relu(bn)
        conv3 = self.conv3(relu)
        sigmoid = torch.sigmoid(conv3)

        return sigmoid

def enc_block(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)))
    return block

def dec_block(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2))
    return block

class Thorn(nn.Module):
    def __init__(self):
        super(Thorn, self).__init__()

        self.conv1 = enc_block(3, 64)
        self.conv2 = enc_block(64, 128)
        self.conv3 = enc_block(128, 256)

        self.att = RFAM(256)

        self.dec1 = dec_block(768, 128)
        self.dec2 = dec_block(384, 64)
        self.dec3 = dec_block(192, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def forward(self, rgb, low, high):

        rgb1 = self.conv1(rgb)
        rgb2 = self.conv2(rgb1)
        rgb3 = self.conv3(rgb2)

        low1 = self.conv1(low)
        low2 = self.conv2(low1 + rgb1)
        low3 = self.conv3(low2 + rgb2)

        high1 = self.conv1(high) # [b, 64, 128, 128]
        high2 = self.conv2(high1 + rgb1) # [b, 128, 64, 64]
        high3 = self.conv3(high2 + rgb2) # [b, 256, 32, 32]

        bridge = self.att(rgb3, low3, high3) #ATT [b, 768, 32, 32] => 256

        dec1 = self.dec1(torch.cat([bridge, low3, high3], dim=1)) # [768,32,32 = > 128, 64, 64]
        dec2 = self.dec2(torch.cat([dec1, low2, high2], dim=1)) # [384,64,64 => 64, 128, 128]
        dec3 = self.dec3(torch.cat([dec2, low1, high1], dim=1)) # [194,128,128 => 32, 256, 256]

        out = self.out(dec3)

        return out


if __name__ == '__main__':
    model = Thorn().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp,inp,inp)
    print('out', out.shape)
