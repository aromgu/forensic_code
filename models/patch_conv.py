import torch.nn.functional as f
import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.xavier_uniform_(m.bias.data)

class patch_conv(nn.Module):
    def __init__(self):
        super(patch_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4= nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
        self.conv5 = nn.Conv2d(32, 16, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
        self.conv6 = nn.Conv2d(16, 1, kernel_size=(3,3), stride=(1, 1), padding=(1,1))

        # self.zero = torch.zeros(batch_size, )

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = f.relu(self.conv6(x))

        return x
