import torch.nn.functional as f
import torch.nn as nn
import torch
# from SRM_filters import get_filters
from utils.wav_pool import DWT, IWT

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.xavier_uniform_(m.bias.data)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        in_ch = 1
        pool = [in_ch, in_ch*4, in_ch*16, in_ch*64, in_ch*256, in_ch*1024]

        self.pool = DWT()
        # self.unpool = IWT()
        self.maxunpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv1.weight = nn.Parameter(get_filters())
        # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        # self.sequential = nn.Sequential(
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
        self.conv7 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # self.unconv5 = nn.ConvTranspose2d(256, 128, kernel_size=(3,3), stride=(2, 2), padding=(1,1), output_padding=(0,0))
        # # self.oneby1 = nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        # self.unconv6 = nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(2, 2), padding=(1,1), output_padding=(0,0))
        # self.unconv7 = nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(2, 2), padding=(1,1), output_padding=(0,0))
        # self.unconv8 = nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=(2, 2), padding=(1,1), output_padding=(0,0))
        #
        # self.unconv9 = nn.ConvTranspose2d(16, 1, kernel_size=(3,3), stride=(2, 2), padding=(1,1), output_padding=(0,0))

        # self.zero = torch.zeros(batch_size, )

    def forward(self, x):
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        x = f.leaky_relu(self.conv5(x))
        x = f.leaky_relu(self.conv6(x))
        x = f.leaky_relu(self.conv7(x))
        x = f.leaky_relu(self.conv8(x))

        return x

        # ll1, hl1, lh1, hh1 = self.pool(x)
        # x = f.relu(self.conv2(torch.cat((ll1, hl1, lh1, hh1),1)))
        # ll2, hl2, lh2, hh2 = self.pool(x)

        # x = f.relu(self.conv3(torch.cat((ll2, hl2, lh2, hh2),1)))
        #ll3, hl3, lh3, hh3 = self.pool(x)

        #x = f.relu(self.conv4(torch.cat((ll3, hl3, lh3, hh3),1)))

# decoder =================

        # unpool = self.maxunpool(x)
        # x = f.relu(self.unconv4(unpool))
        # x = self.oneby1(x)
        #
        # unpool = self.maxunpool(x)
        # x = f.relu(self.unconv5(unpool))
        # x = self.oneby2(x)
        #
        # x = f.relu(self.unconv6(x))
        # x = self.oneby3(x)
        # x = self.unconv7(x)
