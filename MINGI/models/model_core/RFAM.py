import torch
import torch.nn as nn
# patch size = 5

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

class RFAM_2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RFAM_2, self).__init__()
        self.conv1x1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        # self.conv1x1 = nn.Conv2d(ch, ch, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(out_dim, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def forward(self, input):
        # cat = torch.cat((rgb, high),dim=1)
        # convone = self.conv1x1(cat)
        bn = self.bn(input)
        relu = self.relu(bn)
        conv3 = self.conv3(relu)
        sigmoid = torch.sigmoid(conv3)

        return sigmoid

class RFAM_Gate(nn.Module):
    def __init__(self, ch):
        super(RFAM_Gate, self).__init__()
        self.conv1x1 = nn.Conv2d(ch*2, ch, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        # self.conv1x1 = nn.Conv2d(ch, ch, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def forward(self, rgb, gate):
        cat = torch.cat((rgb, gate),dim=1)
        convone = self.conv1x1(cat)
        bn = self.bn(convone)
        relu = self.relu(bn)
        conv3 = self.conv3(relu)
        sigmoid = torch.sigmoid(conv3)

        return sigmoid

if __name__=='__main__':
    model = RFAM(3).cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp, inp, inp)
    print('out', out.shape)
