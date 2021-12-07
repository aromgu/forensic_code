import matplotlib.pyplot as plt

from .Canny import canny
from .ResASPPskip import UNetWithResnet50Encoder

from utils.get_functions import hausdorff
from utils import *
args = get_init()

class Ours(nn.Module):
    def __init__(self, args, device):
        super(Ours, self).__init__()

        self.device = device
        self.preserve_range = args.preserve_range
        self.num_enc = args.num_enc
        self.img_size = args.img_size

        self.fnet = Fnet(args.img_size)
        # self.Unet = unet
        self.region_model = UNetWithResnet50Encoder()
        self.edge_model = UNetWithResnet50Encoder()
        self.Canny = canny

    def forward(self, image, label, criterion, device):
        # Y, U, V = self._RGB2YUV(image) # apply transform only Y channel
        fad_img = self.fnet(image)

        low = torch.zeros(image.size(0),3, args.img_size, args.img_size)
        # for i in range(args.batch_size):
        low[:,0,:,:] = fad_img[:,0,:,:]
        low[:,1,:,:] = fad_img[:,1,:,:]
        low[:,2,:,:] = fad_img[:,2,:,:]

        mid = torch.zeros(image.size(0),3, args.img_size, args.img_size)
        # for i in range(args.batch_size):
        mid[:,0,:,:] = fad_img[:,3,:,:]
        mid[:,1,:,:] = fad_img[:,4,:,:]
        mid[:,2,:,:] = fad_img[:,5,:,:]

        high = torch.zeros(image.size(0),3, args.img_size, args.img_size)
        # for i in range(args.batch_size):
        high[:,0,:,:] = fad_img[:,6,:,:]
        high[:,1,:,:] = fad_img[:,7,:,:]
        high[:,2,:,:] = fad_img[:,8,:,:]

        canny_out = canny(label, device)
        canny_out[canny_out >= 0.5] = 1
        canny_out[canny_out < 0.5] = 0

        # region = torch.cat((mid,low), dim=1)

        low_region_pred = self.region_model(low.to('cuda'))
        mid_region_pred = self.region_model(mid.to('cuda'))

        # plt.imshow(canny_out[2].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        # plt.tight_layout()
        # plt.axis('off')
        # plt.show()
        high_edge_pred = self.edge_model(high.to('cuda'))
        mid_edge_pred = self.edge_model(mid.to('cuda'))

        high_edge_pred = torch.sigmoid(high_edge_pred)
        mid_edge_pred = torch.sigmoid(mid_edge_pred)

        # aver_edge_loss = 0.0
        # for gt_, pred_ in zip(canny_out, high_pred) :
        #     gt_ = gt_.squeeze()
        #     pred_ = pred_.squeeze()
        #     edge_loss = averaged_hausdorff_distance(gt_, pred_)
        #     aver_edge_loss += edge_loss
        # aver_edge_loss /= canny_out.size(0)
        # print('hausloss', aver_edge_loss)
        return canny_out, low_region_pred, mid_region_pred, high_edge_pred, mid_edge_pred

    # def _RGB2YUV(self, img):
    #     YUV = k_color.rgb_to_yuv(img)
    #
    #     return YUV[..., 0, :, :].unsqueeze(dim=1), YUV[..., 1, :, :], YUV[..., 2, :, :]
    #
    # def _YUV2RGB(self, img):
    #     RGB = k_color.yuv_to_rgb(img)
    #
    #     return RGB

