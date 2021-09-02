import sys

from .Unet import *
from .cnnwav import CNN
from .MGP import extract
from .Canny import canny
from .Ours import Ours

def get_model(model_name) :
    if model_name == 'Unet' :
        model = Unet()
    elif model_name == 'CNN' :
        model = CNN()
    elif model_name == 'Ours':
        model = Ours()

    else :
        NotImplementedError("You can use only Unet")
        sys.exit(-1)

    return model

