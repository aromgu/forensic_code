import torch.nn.functional as f
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channel):
        super(CNN, self).__init__()
        pool = [input_channel, input_channel*4, input_channel*16, input_channel*64, input_channel*256, input_channel*1024]
        # self.unpool = IWT()

        self.maxunpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv1.weight = nn.Parameter(get_filters())
        # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        # self.sequential = nn.Sequential(
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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
        x = self.conv8(x)

        return x