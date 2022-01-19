import torch
import torch.nn as nn

def Conv_block1_3():
    block = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
    return block

def Conv_block4(in_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
    return block

def Conv_block6_7():
    block = nn.Sequential(
        nn.Conv2d(512, 4096, kernel_size=(7,7), stride=(1,1), padding=(1,1)),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Conv2d(4096, 2, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        nn.Upsample(scale_factor=2)
    )
    return block

class MFCN(nn.Module):
    def __init__(self):
        super(MFCN, self).__init__()
        self.conv1_3 = Conv_block1_3()
        self.conv4 = Conv_block4(256)
        self.conv5 = Conv_block4(512)
        self.conv6_7 = Conv_block6_7()
        self.upspl2 = nn.Upsample(scale_factor=2)
        self.upspl8 = nn.Upsample(scale_factor=8)
        self.crop3_t = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
        self.crop3_b = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
        self.crop4_t = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
        self.crop4_b = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))

        self.regionout = nn.Conv2d(2, 1, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        self.edgeout = nn.Conv2d(2, 1, kernel_size=(1,1), stride=(1,1), dilation=(1,1))

    def forward(self, x):
        conv1_3 = self.conv1_3(x) # [b, 256, 32, 32]
        conv4 = self.conv4(conv1_3) # [b, 512, 16, 16]
        conv5 = self.conv5(conv4) # [b, 512, 8, 8]
        conv6_7 = self.conv6_7(conv5) # [b, 2, 16, 16]

        crop3_t = self.crop3_t(conv1_3) # [b, 2, 32, 32]
        crop3_b = self.crop3_b(conv1_3) # [b, 2, 32, 32]
        crop4_t = self.crop4_t(conv4) # [b, 2, 16, 16]
        crop4_b = self.crop4_b(conv4) # [b, 2, 16, 16]

        in_t = crop4_t + conv6_7
        in_b = crop4_b + conv6_7
        in_t = self.upspl2(in_t)
        in_b = self.upspl2(in_b)

        out_t = crop3_t + in_t
        out_b = crop3_b + in_b
        region_out = self.upspl8(out_t) # out_t
        edge_out = self.upspl8(out_b) # out_b

        region_out = self.regionout(region_out)
        edge_out = self.edgeout(edge_out)

        return region_out, edge_out

if __name__ == '__main__':
    model = MFCN().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    region, edge = model(inp)
    print('out', edge.shape)
