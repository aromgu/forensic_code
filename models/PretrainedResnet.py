import torch.nn.functional as f
import torch.nn as nn
import torchvision.models as models

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()

        modules = list(models.resnet18(pretrained=True).children())[:-3]
        # delete maxpooling
        del modules[3]

        self.feature_extractor = nn.Sequential(*modules) # => DenseNet
        for layer in self.feature_extractor[-1:]:
            layer.trainable = True

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn5 = nn.BatchNorm2d(64)

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn6 = nn.BatchNorm2d(32)

        self.out = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = f.leaky_relu(self.feature_extractor(x))
        x = f.leaky_relu(self.bn4(self.conv4(self.up4(x))))
        x = f.leaky_relu(self.bn5(self.conv5(self.up5(x))))
        x = f.leaky_relu(self.bn6(self.conv6(self.up6(x))))
        out = self.out(x)
        return out