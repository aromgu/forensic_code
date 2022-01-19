import torch.nn as nn

class RCU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RCU, self).__init__()
        self.rcu = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    def forward(self, x):
        identity = self.rcu(x)
        out = x + identity
        return out

class Multi_fuse(nn.Module):
    def __init__(self, out_dim, factor):
        super(Multi_fuse, self).__init__()
        self.scale2 = nn.Sequential(
            nn.Conv2d(64, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=factor)
        )

        self.no_scale = nn.Sequential(
            nn.Conv2d(64, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        )
        self.scale4 = nn.Sequential(
            nn.Conv2d(128, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=factor*1)
        )
        self.scale8 = nn.Sequential(
            nn.Conv2d(256, out_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Upsample(scale_factor=factor*2)
        )
    def forward(self, x128, x64, x32):
        if x128 == 'None':
            sc2 = self.scale4(x64)
            sc3 = self.scale8(x32)
            out = sc2 + sc3
            return out
        else:
            sc1 = self.no_scale(x128)
            sc2 = self.scale4(x64)
            sc3 = self.scale8(x32)
            out = sc1 + sc2 + sc3
            return out


class Chain_pool(nn.Module):
    def __init__(self, in_dim):
        super(Chain_pool, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.Sequential(
            nn.MaxPool2d((5,5), stride=1, padding=(2,2)),
            nn.Conv2d(in_dim, in_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        )

    def forward(self, x):
        x = self.relu(x)
        o1 = self.pool(x)
        o2 = self.pool(o1)
        o3 = self.pool(o2)

        out = o1 + o2 + o3
        return out

# class Refine(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(Refine, self).__init__()
#         self.rcu = RCU(in_dim, out_dim)
#         self.fusion = Multi_fuse(out_dim)
#         self.chain = Chain_pool(in_dim, out_dim)
#
#     def forward(self, x):
#         x = self.rcu(x)
#         x = self.fusion(x)
#         out = self.chain(x)
#
#         return out