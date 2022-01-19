import sys
from utils import *
from utils.config import config
from .res50 import ResASPP50
from .res18_base import Res18rfam
from .res18 import ResFeat
from .vgg16 import VGGFeat
from .resnext101 import resnextFeat
from .vgg16_seg import VGGseg
from .resnet101 import Res101
from .res50_seg import Res50_seg
from .res34_seg import Res34_seg
from .res18_gate import ResGate
from .res18worms import Res18worms
from .SE_Net import SENet
from .Unet import Unet
from .DeeplabV3plus import DeepLabV3Plus
from .Segnet import SegNet
from .MFCN import MFCN
from .DCUNet import DCUNet
from .RRUNet import RRUNet
from .ThornNet import Thorn

def get_model(model_name, args=None, device= 'cuda') :
    if model_name == 'res18_base':
        model = Res18rfam()
    elif model_name == 'res18':
        model = ResFeat()
    elif model_name == 'res18gate':
        model = ResGate()
    elif model_name == 'res18worms':
        model = Res18worms()
    elif model_name == 'res34_seg':
        model = Res34_seg()
    elif model_name == 'res50':
        model = ResASPP50(3, 'resnet50', False, False, multi_grid=True, multi_dilation=[4, 8, 16])
    elif model_name == 'res50_seg':
        model = Res50_seg()
    elif model_name == 'res101':
        model = Res101()
    elif model_name == 'resnext101':
        model = resnextFeat()
    elif model_name == 'vgg16':
        model = VGGFeat()
    elif model_name == 'vgg16_seg':
        model = VGGseg()
    elif model_name == 'SE':
        model = SENet()
    elif model_name == 'Unet' :
        model = Unet()
    elif model_name == 'DeepLabV3+' :
        model = DeepLabV3Plus(
            n_classes=1,
            n_blocks=[3, 4, 23, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16)
    elif model_name == 'SegNet' :
        model = SegNet(input_nbr=3,label_nbr=1)
    elif model_name == 'MFCN':
        model = MFCN()
    elif model_name == 'DCUNET':
        model = DCUNet()
    elif model_name == 'RRUNET':
        model = RRUNet(3)
    elif model_name == 'thorn':
        model = Thorn()

    else :
        NotImplementedError("You can use only Unet")
        sys.exit(-1)

    return model

