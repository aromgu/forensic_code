import sys

from .Unet import *
from .cnnwav import CNN
from .Canny import canny
from .Ours import Ours

def get_model(model_name, args=None, device='cpu') :
    if model_name == 'Unet' :
        model = Unet()
    elif model_name == 'CNN' :
        model = CNN()
    elif model_name == 'Ours':
        model = Ours(args, device)

    else :
        NotImplementedError("You can use only Unet")
        sys.exit(-1)

    return model

