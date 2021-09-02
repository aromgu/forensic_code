import torch.nn as nn
import torch.nn.functional as F

class patch_conv(nn.Module):
    def __init__(self):
        super(patch_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4= nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
        self.conv5 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
        self.conv6 = nn.Conv2d(16, 1, kernel_size=(3,3), stride=(1, 1), padding=(1,1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        return x
