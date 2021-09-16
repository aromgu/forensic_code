import sys

from .Unet import *
from .PretrainedResnet import ResCNN
from .Canny import canny
from .Ours_MGP import Ours
from .RRUNet import Ringed_Res_Unet
from .CNN import CNN
from .PretrainedASPP import Block

def get_model(model_name, args=None, device='cpu') :
    if model_name == 'Unet' :
        model = Unet()
    elif model_name == 'Ours':
        model = Ours(args, device)
    elif model_name == 'RRU':
        model = Ringed_Res_Unet(3)
    elif model_name == 'ResCNN' :
        model = ResCNN()
    elif model_name == 'ResASPP':
        model = Block()


    else :
        NotImplementedError("You can use only Unet")
        sys.exit(-1)

    return model

