import torch
import torch.nn as nn

import kornia.filters.sobel as sobel_filter

class DilationConv(nn.Module) :
    def __init__(self, input_channels, output_channels):
        super(DilationConv, self).__init__()

        self.dilation_conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.dilation_conv_2 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        self.dilation_conv_3 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=(3, 3))

    def forward(self, x):
        output_rate1 = self.dilation_conv_1(x)
        output_rate2 = self.dilation_conv_2(x)
        output_rate3 = self.dilation_conv_3(x)

        output = torch.cat([output_rate1, output_rate2, output_rate3], dim=1)

        return output

class ResidualConv(nn.Module) :
    def __init__(self, input_channels, output_channels, act_fn):
        super(ResidualConv, self).__init__()

        self.residual1 = conv_block(input_channels, output_channels, act_fn)
        self.residual2 = nn.Sequential(
            conv_block(output_channels, output_channels, act_fn),
            conv_block(output_channels, output_channels, act_fn),
        )
        self.residual3 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        residual = self.residual1(x)
        output   = self.residual2(residual)
        output   = output + residual
        output   = self.residual3(output)

        return output

class DCUNet(nn.Module) :
    def __init__(self, input_channels=3, num_classes=1):
        super(DCUNet, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.region_num_filters = 64
        self.residual_num_filters = 8

        act_fn = nn.ReLU(inplace=True)
        self.maxpool = maxpool()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        ############ REGION STREAM ARCHITECTURE ############
        self.region_encoder1 = nn.Sequential(
            conv_block(self.input_channels, self.region_num_filters * 1, act_fn),
            conv_block(self.region_num_filters * 1, self.region_num_filters * 1, act_fn)
        )

        self.region_encoder2 = nn.Sequential(
            conv_block(self.region_num_filters * 1, self.region_num_filters * 2, act_fn),
            conv_block(self.region_num_filters * 2, self.region_num_filters * 2, act_fn)
        )

        self.region_encoder3 = nn.Sequential(
            conv_block(self.region_num_filters * 2, self.region_num_filters * 4, act_fn),
            conv_block(self.region_num_filters * 4, self.region_num_filters * 4, act_fn),
            conv_block(self.region_num_filters * 4, self.region_num_filters * 4, act_fn)
        )

        self.region_encoder4 = nn.Sequential(
            conv_block(self.region_num_filters * 4, self.region_num_filters * 8, act_fn),
            conv_block(self.region_num_filters * 8, self.region_num_filters * 8, act_fn),
            conv_block(self.region_num_filters * 8, self.region_num_filters * 8, act_fn)
        )

        self.region_encoder5 = nn.Sequential(
            conv_block(self.region_num_filters * 8, self.region_num_filters * 8, act_fn),
            conv_block(self.region_num_filters * 8, self.region_num_filters * 8, act_fn),
            conv_block(self.region_num_filters * 8, self.region_num_filters * 8, act_fn)
        )
        ############ REGION STREAM ARCHITECTURE ############

        ############ RESIDUAL STREAM ARCHITECTURE ############
        self.residual_encoder1 = ResidualConv(self.input_channels, self.residual_num_filters * 1, act_fn)
        self.residual_encoder2 = ResidualConv(self.residual_num_filters * 1, self.residual_num_filters * 2, act_fn)
        self.residual_encoder3 = ResidualConv(self.residual_num_filters * 2, self.residual_num_filters * 4, act_fn)
        self.residual_encoder4 = ResidualConv(self.residual_num_filters * 4, self.residual_num_filters * 8, act_fn)
        self.residual_encoder5 = ResidualConv(self.residual_num_filters * 8, self.residual_num_filters * 16, act_fn)
        ############ RESIDUAL STREAM ARCHITECTURE ############

        self.dilation_conv = DilationConv(640, 256)

        ############ DECODER STREAM ARCHITECTURE ############
        self.decoder1 = nn.Sequential(
            nn.Conv2d(1280, self.region_num_filters * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(768, self.region_num_filters * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(384, self.region_num_filters * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2)
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(192, self.region_num_filters * 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2)
        )
        ############ DECODER STREAM ARCHITECTURE ############

        ############ OUTPUT STREAM ARCHITECTURE ############
        self.output = nn.Sequential(
            nn.Conv2d(self.region_num_filters * 1, self.region_num_filters * 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(self.region_num_filters * 1, self.num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        residual_x = sobel_filter(x)

        ############ REGION STREAM ############
        region_output1_ = self.region_encoder1(x)               # [B, 3, 256, 256] => [B, 64, 256, 256]
        region_output1  = self.maxpool(region_output1_)         # [B, 64, 256, 256] => [B, 64, 128, 128]

        region_output2_ = self.region_encoder2(region_output1)  # [B, 64, 128, 128] => [B, 128, 128, 128]
        region_output2  = self.maxpool(region_output2_)         # [B, 128, 128, 128] => [B, 128, 64, 64]

        region_output3_ = self.region_encoder3(region_output2)  # [B, 128, 64, 64] => [B, 256, 64, 64]
        region_output3  = self.maxpool(region_output3_)         # [B, 256, 64, 64] => [B, 256, 32, 32]

        region_output4_ = self.region_encoder4(region_output3)  # [B, 256, 32, 32] => [B, 512, 32, 32]
        region_output4  = self.maxpool(region_output4_)         # [B, 512, 32, 32] => [B, 512, 16, 16]

        region_output5_ = self.region_encoder5(region_output4)  # [B, 512, 16, 16] => [B, 512, 16, 16]
        region_output5  = self.maxpool(region_output5_)         # [B, 512, 16, 16] => [B, 512, 8, 8]

        ############ REGION STREAM ############
        residual_output1 = self.residual_encoder1(residual_x)       # [B, 3, 256, 256] => [B, 8, 128, 128]
        residual_output2 = self.residual_encoder2(residual_output1) # [B, 8, 128, 128] => [B, 16, 64, 64]
        residual_output3 = self.residual_encoder3(residual_output2) # [B, 16, 64, 64] => [B, 32, 32, 32]
        residual_output4 = self.residual_encoder4(residual_output3) # [B, 32, 32, 32] => [B, 64, 16, 16]
        residual_output5 = self.residual_encoder5(residual_output4) # [B, 64, 16, 16] => [B, 128, 8, 8]
        ############ RESIDUAL STREAM ############

        concat_feature = torch.cat([region_output5, residual_output5], dim=1) # [B, 512, 8, 8] + [B, 128, 8, 8] => [B, 640, 8, 8]
        dilation_output = self.dilation_conv(concat_feature)                  # [B, 640, 8, 8] => [B, 768, 8, 8]
        upsample_output = self.upsample(dilation_output)                      # [B, 768, 8, 8] => [B, 768, 16, 16]

        ############ DECODER STREAM ############
        decoder_output_cat1 = torch.cat([upsample_output, region_output4], dim=1) # [B, 768, 16, 16] + [B, 512, 16, 16] => [B, 1280, 16, 16]
        decoder_output1     = self.decoder1(decoder_output_cat1)                  # [B, 1280, 16, 16] => [B, 512, 32, 32]

        decoder_output_cat2 = torch.cat([decoder_output1, region_output3], dim=1) # [B, 512, 32, 32] + [B, 256, 32, 32] => [B, 768, 32, 32]
        decoder_output2     = self.decoder2(decoder_output_cat2)                  # [B, 768, 32, 32] => [B, 256, 64, 64]

        decoder_output_cat3 = torch.cat([decoder_output2, region_output2], dim=1) # [B, 256, 64, 64] + [B, 128, 64, 64] => [B, 384, 64, 64]
        decoder_output3     = self.decoder3(decoder_output_cat3)                  # [B, 384, 64, 64] => [B, 128, 128, 128]

        decoder_output_cat4 = torch.cat([decoder_output3, region_output1], dim=1) # [B, 128, 128, 128] + [B, 64, 128, 128] => [B, 192, 128, 128]
        decoder_output4     = self.decoder4(decoder_output_cat4)                  # [B, 192, 128, 128] => [B, 64, 256, 256]
        ############ DECODER STREAM ############

        ############ OUTPUT STREAM ############
        final_output = self.output(decoder_output4) # [B, 64, 256, 256] => [B, 1, 256, 256]
        ############ OUTPUT STREAM ############

        return final_output

def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim)
    )

    return model

def maxpool() :
    pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    return pool

def conv_block(in_dim, out_dim, act_fn) :
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        act_fn
    )

    return model


if __name__=="__main__" :
    input_data = torch.rand((2, 3, 256, 256)).cuda()
    model = DCUNet().cuda()
    output = model(input_data)
    print(output.shape)
