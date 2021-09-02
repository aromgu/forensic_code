import os

import torch

from utils.plot_functions import plot_loss

def save_best_model(parent_dir, epoch, model, model_name, optimizer, loss, fad_option):
    PATH = os.path.join(parent_dir, model_name, 'fad_{}'.format(fad_option))
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
    }, os.path.join(PATH, "best_{}.pth".format(epoch)))
    print(f'saved best model : epoch{epoch}')

def save_last_model(parent_dir, epoch, model, model_name, fad_option):
    PATH = os.path.join(parent_dir,model_name,f'fad_{fad_option}')
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(model, os.path.join(PATH, "last_{}.pth".format(epoch)))
    print(f'saved last model : epoch{epoch}')

def save_history(history, parent_dir, fad_option, model_name):
    plot_loss(history, model_name, fad_option)
    f = open(os.path.join(parent_dir, model_name, f'fad_{fad_option}','history.txt'), 'w')
    f.write(str(history))
    f.close()