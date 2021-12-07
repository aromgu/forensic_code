import sys

from .Unet import *
from .PretrainedResnet import ResCNN
from .Canny import canny
from .Ours_FAD import Ours
from .RRUNet import Ringed_Res_Unet
from .CNN import CNN
from .base_fusion_element import Elementfusion
from .base_fusion_pointwiseconv import DUAL
from .base_fusion import BaseNet
from .base_fusion_dfmbfi import Dfmbfi
from .base_fusion_rfam import RFAMNET
from .base_fusion_rfam_uprgb import RFAMUPRGB
from .DeeplabV3plus import DeepLabV3Plus
from .Segnet import SegNet_
from .DenseASPP import DenseASPP_
from .Wavesnet import SegNet_VGG
from .ResASPPskip import UNetWithResnet50Encoder
from .base_fusion_wavelet import Wavfusion
from .Cat_Net import get_seg_model
from utils import *
from utils.config import config
from .Mantranet import MantraNet
from .base_fusion_oneencoder import OneEncoder
from .PretrainedASPP import ResASPP
from .ResASPP50 import ResASPP50
from .res18rfam import Res18rfam
from .res18_fuse_autp import res18autp
from .transforensics import transforensics

# args = argparse.Namespace(cfg='utils/CAT_full.yaml', local_rank=0, opts=None)
# device = get_device()

def get_model(model_name, args=None, device= 'cuda') :
    if model_name == 'Unet' :
        model = Unet()
    elif model_name == 'Ours':
        model = Ours(args, device)
    elif model_name == 'RRU':
        model = Ringed_Res_Unet(3)
    elif model_name == 'ResCNN' :
        model = ResCNN()
    elif model_name == 'elementmul':
        model = Elementfusion(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'rgbadd':
        model = BaseNet(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'dfmbfi':
        model = Dfmbfi(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'wavfusion':
        model = Wavfusion(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'rfam':
        model = RFAMNET(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'rfamuprgb':
        model = RFAMUPRGB(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'DUAL':
        model = DUAL(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'Deep3plus':
        model = DeepLabV3Plus(
        n_classes=1,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
    )
    elif model_name == 'Segnet':
        model = SegNet_(input_nbr=3, label_nbr=1)
    elif model_name == 'DenseASPP':
        model = DenseASPP_()
    elif model_name == 'Wavesnet':
        model = SegNet_VGG(2, 'haar')
    elif model_name == 'oneencoder':
        model = OneEncoder(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'resaspp18':
        model = ResASPP()
    elif model_name == 'catnet':
        model = get_seg_model(config)
    elif model_name == 'mantranet':
        model = MantraNet(device)
    elif model_name == 'resaspp50':
        model = ResASPP50(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'res18rfam':
        model = Res18rfam()
    elif model_name == 'res18autp':
        model = res18autp()
    elif model_name == 'transforensics':
        model = transforensics()
    else :
        NotImplementedError("You can use only Unet")
        sys.exit(-1)

    return model

