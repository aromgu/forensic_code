import torch
import torch.nn as nn
# patch size = 5

class RFAM(nn.Module):
    def __init__(self, ch):
        super(RFAM, self).__init__()
        # self.conv1x1 = nn.Conv2d(ch*3, ch, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        self.conv1x1 = nn.Conv2d(ch*2, ch, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    # def forward(self, rgb, low, high):
    #     cat = torch.cat((rgb, low, high),dim=1)

    def forward(self, tp, au):
        cat = torch.cat((tp, au), dim=1)
        convone = self.conv1x1(cat)
        bn = self.bn(convone)
        relu = self.relu(bn)
        conv3 = self.conv3(relu)
        sigmoid = torch.sigmoid(conv3)
        # b,ch,w,h = sigmoid.size()
        # rgb = sigmoid[:,:int(ch/3),:,:]
        # low = sigmoid[:,int(ch/3):int((ch/3)*2),:,:]
        # high = sigmoid[:,int(ch/3*2):,:,:]
        return sigmoid


if __name__=='__main__':
    model = RFAM(3).cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    out = model(inp, inp, inp)
    print('out', out.shape)
