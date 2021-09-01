import sys

from .Unet import *
from .patch_conv import *
from .cnnwav import CNN

def get_model(model_name) :
    if model_name == 'Unet' :
        model = Unet()
    elif model_name == 'CNN' :
        model = CNN()
    else :
        NotImplementedError("You can use only Unet")
        sys.exit(-1)

    return model

def get_patch_conv():
    Patch_conv = patch_conv()
    return Patch_conv
