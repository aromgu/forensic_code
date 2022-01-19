# Multi-task SE-Network for Image Splicing Localization
import torch
import torch.nn as nn

class Unet(nn.Module) :
    def __init__(self, in_dim=3, n_class=3, num_filters=32):
        super(Unet, self).__init__()

        self.in_dim      = in_dim
        self.n_class     = n_class
        self.num_filters = num_filters

        act_fn = nn.LeakyReLU(0.2, inplace=True)

        # Encoding Parts
        self.down1 = conv_block_2(in_dim, self.num_filters, act_fn)
        self.pool1 = maxpool(self.num_filters, self.num_filters)
        self.down2 = conv_block_2(self.num_filters * 1, self.num_filters * 2, act_fn)
        self.pool2 = maxpool(self.num_filters*2, self.num_filters*2)
        self.down3 = conv_block_2(self.num_filters * 2, self.num_filters * 4, act_fn)
        self.pool3 = maxpool(self.num_filters*4, self.num_filters*4)
        self.down4 = conv_block_2(self.num_filters * 4, self.num_filters * 8, act_fn)
        self.pool4 = maxpool(self.num_filters*8, self.num_filters*8)

        self.bridge = conv_block_2(self.num_filters * 8, self.num_filters * 16, act_fn)

        # Decoding Parts
        self.trans1 = up_spl(self.num_filters * 16, self.num_filters * 8)
        self.up1    = conv_block_2(self.num_filters *16, self.num_filters * 8, act_fn)
        self.trans2 = up_spl(self.num_filters * 8, self.num_filters * 4)
        self.up2    = conv_block_2(self.num_filters * 8, self.num_filters * 4, act_fn)
        self.trans3 = up_spl(self.num_filters * 4, self.num_filters * 2)
        self.up3    = conv_block_2(self.num_filters * 4, self.num_filters * 2, act_fn)
        self.trans4 = up_spl(self.num_filters * 2, self.num_filters * 1)
        self.up4    = conv_block_2(self.num_filters * 2, self.num_filters * 1, act_fn)

        # output block
        self.out = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        down1 = self.down1(x)
        pool1 = self.pool1(down1)
        down2 = self.down2(pool1)
        pool2 = self.pool2(down2)
        down3 = self.down3(pool2)
        pool3 = self.pool3(down3)
        down4 = self.down4(pool3)
        pool4 = self.pool4(down4)

        bridge = self.bridge(pool4)

        # feature decoding
        trans1  = self.trans1(bridge)
        concat1 = torch.cat([trans1, down4], dim=1)
        up1     = self.up1(concat1)
        trans2  = self.trans2(up1)
        concat2 = torch.cat([trans2, down3], dim=1)

        up2     = self.up2(concat2)
        trans3  = self.trans3(up2)
        concat3 = torch.cat([trans3, down2], dim=1)
        up3     = self.up3(concat3)
        trans4  = self.trans4(up3)
        concat4 = torch.cat([trans4, down1], dim=1)
        up4     = self.up4(concat4)

        out = self.out(up4)
        return out

def conv_block_2(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim))
    return model

def up_spl(in_dim, out_dim) :
    model = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(in_dim, out_dim, kernel_size=(1,1), stride=(1,1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())
    return model

def maxpool(in_dim, out_dim) :
    pool = nn.Conv2d(in_dim, out_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    return pool

def conv_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        act_fn)
    return model

class FAL(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FAL, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1,1), stride=(1,1), bias=False),
            nn.ReLU())
    def forward(self, x):
        identity = self.conv(x)
        out = x + identity
        return out

class SEAM(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, out_ch, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SEAM, self).__init__()
        num_channels_reduced = out_ch // reduction_ratio
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=(1,1), stride=(1,1))
        # self.original = nn.Conv2d(num_channels, num_channels, kernel_size=(1,1), stride=(1,1), dilation=(1,1))
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, out_ch, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        original = self.conv(input_tensor)

        batch_size, num_channels, H, W = original.size()
        # Average along each channel
        # conv = self.conv(input_tensor)

        squeeze_tensor = original.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(original, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.edge_stream = Unet()
        self.label_mask_stream = Unet()
        self.seam1 = SEAM(256, 256)
        self.seam2 = SEAM(128, 128)
        self.seam3 = SEAM(64, 64)
        self.seam4 = SEAM(32, 32)
        self.fal1 = FAL(32, 32)
        self.fal2 = FAL(64, 64)
        self.fal3 = FAL(128, 128)
        self.fal4 = FAL(256, 256)
        self.fal5 = FAL(512, 512)

        self.low_level1 = nn.Conv2d(3, 256, kernel_size=(8,8), stride=(8,8))
        self.low_level2 = nn.Conv2d(3, 128, kernel_size=(4,4), stride=(4,4))
        self.low_level3 = nn.Conv2d(3, 64, kernel_size=(2,2), stride=(2,2))
        self.low_level4 = nn.Conv2d(3, 32, kernel_size=(1,1), stride=(1,1))

        self.label_mask_out = nn.Conv2d(32, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1))


    def forward(self, x):
        # EDGE STREAM
        edge1 = self.edge_stream.down1(x) # [b, 32, 256, 256]
        edge1_ = self.edge_stream.pool1(edge1) # [b, 32, 128, 128]
        fal1 = self.fal1(edge1_)

        edge2 = self.edge_stream.down2(edge1_) # [b, 64, 128, 128]
        edge2_ = self.edge_stream.pool2(edge2) # [b, 64, 64, 64]
        fal2 = self.fal2(edge2_)

        edge3 = self.edge_stream.down3(edge2_) # [b, 128, 64, 64]
        edge3_ = self.edge_stream.pool3(edge3) # [b, 128, 32, 32]
        fal3 = self.fal3(edge3_)

        edge4 = self.edge_stream.down4(edge3_) # [b, 256, 32, 32]
        edge4_ = self.edge_stream.pool4(edge4) # [b, 256, 16, 16]
        fal4 = self.fal4(edge4_)

        edge_bridge = self.edge_stream.bridge(edge4_) # [b, 512, 16, 16]
        fal5 = self.fal5(edge_bridge)

        edge5 = self.edge_stream.trans1(edge_bridge)
        edge5 = self.edge_stream.up1(torch.cat([edge5, edge4], dim=1)) # [2, 256, 32, 32]
        fal6 = self.fal4(edge5)

        edge6 = self.edge_stream.trans2(edge5)
        edge6 = self.edge_stream.up2(torch.cat([edge3, edge6], dim=1)) # [2, 128, 64, 64]
        fal7 = self.fal3(edge6)

        edge7 = self.edge_stream.trans3(edge6)
        edge7 = self.edge_stream.up3(torch.cat([edge2, edge7], dim=1)) # [2, 64, 128, 128]
        fal8 = self.fal2(edge7)

        edge8 = self.edge_stream.trans4(edge7)
        edge8 = self.edge_stream.up4(torch.cat([edge1, edge8], dim=1)) # [2, 32, 256, 256]

        edge_out = self.edge_stream.out(edge8)

        # LABEL MASK STREAM =============
        label1 = self.label_mask_stream.down1(x) # [b, 32, 256, 256]
        label1_ = self.label_mask_stream.pool1(label1) # [b, 32, 128, 128]
        label1_ = torch.add(label1_, fal1)

        label2 = self.label_mask_stream.down2(label1_) # [b, 64, 128, 128]
        label2_ = self.label_mask_stream.pool2(label2) # [b, 64, 64, 64]
        label2_ = torch.add(label2_,fal2)

        label3 = self.label_mask_stream.down3(label2_) # [b, 128, 64, 64]
        label3_ = self.label_mask_stream.pool3(label3) # [b, 128, 32, 32]
        label3_ = torch.add(label3_, fal3)

        label4 = self.label_mask_stream.down4(label3_) # [b, 256, 32, 32]
        label4_ = self.label_mask_stream.pool4(label4) # [b, 256, 16, 16]
        label4_ = torch.add(label4_, fal4)

        label_bridge = self.label_mask_stream.bridge(label4_) # [b, 512, 16, 16]
        label_bridge = torch.add(label_bridge, fal5)

        low1 = self.low_level1(x)
        label5 = self.label_mask_stream.trans1(label_bridge) # [b, 256, 32, 32]
        label5 = torch.add(low1, label5)
        seam1 = self.seam1(label5)
        label5 = self.label_mask_stream.up1(torch.cat([seam1, label4], dim=1)) # [2, 256, 32, 32]
        label5 = torch.add(label5, fal6)

        low2 = self.low_level2(x)
        label6 = self.label_mask_stream.trans2(label5)
        label6 = torch.add(low2, label6)
        seam2 = self.seam2(label6)
        label6 = self.label_mask_stream.up2(torch.cat([label3, seam2], dim=1)) # [2, 128, 64, 64]
        label6 = torch.add(label6, fal7)

        low3 = self.low_level3(x)
        label7 = self.label_mask_stream.trans3(label6)
        label7 = torch.add(low3, label7)
        seam3 = self.seam3(label7)
        label7 = self.label_mask_stream.up3(torch.cat([label2, seam3], dim=1)) # [2, 64, 128, 128]
        label7 = torch.add(label7, fal8)

        low4 = self.low_level4(x)
        label8 = self.label_mask_stream.trans4(label7)
        label8 = torch.add(low4, label8)
        seam4 = self.seam4(label8)
        label8 = self.label_mask_stream.up4(torch.cat([label1, seam4], dim=1)) # [2, 32, 256, 256]

        mask_edges = self.label_mask_stream.out(label8)
        label_mask = self.label_mask_out(label8)

        return edge_out, mask_edges, label_mask

if __name__ == '__main__':
    model = SENet().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    o1 = model(inp)
