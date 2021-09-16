import torch.nn.functional as f
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(512)

        self.unconv6 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), output_padding=(1,1),padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(256)
        self.unconv7 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), output_padding=(1,1),padding=(1, 1))
        self.bn7 = nn.BatchNorm2d(128)
        self.unconv8 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), output_padding=(1,1),padding=(1, 1))
        self.bn8 = nn.BatchNorm2d(64)
        self.unconv9 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), output_padding=(1,1),padding=(1, 1))
        self.bn9 = nn.BatchNorm2d(32)
        self.unconv10 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), output_padding=(1,1), padding=(1, 1))
        self.bn10 = nn.BatchNorm2d(16)

        self.out = nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        # x = f.leaky_relu(self.feature_extractor(x))
        x = f.leaky_relu(self.bn1(self.conv1(x)))
        x = f.leaky_relu(self.bn2(self.conv2(x)))
        x = f.leaky_relu(self.bn3(self.conv3(x)))
        x = f.leaky_relu(self.bn4(self.conv4(x)))
        x = f.leaky_relu(self.bn5(self.conv5(x)))
        x = f.leaky_relu(self.bn6(self.unconv6(x)))
        x = f.leaky_relu(self.bn7(self.unconv7(x)))
        x = f.leaky_relu(self.bn8(self.unconv8(x)))
        x = f.leaky_relu(self.bn9(self.unconv9(x)))
        x = f.leaky_relu(self.bn10(self.unconv10(x)))

        out = self.out(x)
        return out