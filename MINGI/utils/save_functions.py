import os

import torch

from utils.plot_functions import plot_loss
from utils.get_functions import print_configurations

def save_last_model(parent_dir, epoch, model, optimizer, save_root):
    PATH = os.path.join(parent_dir,save_root)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    check_point = {
        'model': model.module if torch.cuda.device_count() > 1 else model,
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(check_point, os.path.join(PATH, 'last_{}.pth'.format(epoch)))

    print(f'saved last model : epoch{epoch}')

    # torch.save(model.module.state_dict(), os.path.join(PATH, "last_{}.pth".format(epoch)))
    # torch.save(check_point, os.path.join(PATH, 'lase_{}.pth'.format(epoch)))
        # torch.save(model.state_dict(), os.path.join(PATH, "last_{}.pth".format(epoch)))

def save_best_model(parent_dir, epoch, model, optimizer, save_root):
    PATH = os.path.join(parent_dir,save_root)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    check_point = {
        'model': model.module if torch.cuda.device_count() > 1 else model,
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(check_point, os.path.join(PATH, 'last_{}.pth'.format(epoch)))

    print(f'saved last model : epoch{epoch}')
def save_history(history, parent_dir, save_root):
    plot_loss(parent_dir, history, save_root)
    f = open(os.path.join(parent_dir, save_root)+'/history.txt', 'w')
    f.write(str(history))
    f.close()

def save_metrics(args, parent_dir, save_root, metrcis, save_path) :
    print("+++++++++++ TEST REPORT +++++++++++")
    print("AUC : {}\n".format(metrcis[0]))
    print("IoU : {}\n".format(metrcis[1]))
    print("F1-score : {}\n".format(metrcis[2]))
    print("Precision : {}\n".format(metrcis[3]))
    print("Recall : {}\n".format(metrcis[4]))
    print("+++++++++++ TEST REPORT +++++++++++")

    print_configurations(args)

    f = open(os.path.join(parent_dir, save_root, save_path), 'w')

    f.write("###################### TEST REPORT ######################\n")
    f.write("AUC        : {}\n".format(metrcis[0]))
    f.write("IoU        : {}\n".format(metrcis[1]))
    f.write("F1-score   : {}\n".format(metrcis[2]))
    f.write("Precision  : {}\n".format(metrcis[3]))
    f.write("Recall     : {}\n".format(metrcis[4]))
    f.write("###################### TEST REPORT ######################\n")

    f.write("+++++++++++++++++++++++++++++++++++")
    f.write("data_type      : {}\n".format(args.dataloader))
    f.write("model_name     : {}\n".format(args.model_name))
    f.write("optimizer      : {}\n".format(args.optimizer))
    f.write("learning_rate  : {}\n".format(args.learning_rate))
    f.write("momentum       : {}\n".format(args.momentum))
    f.write("weight_decay   : {}\n".format(args.weight_decay))
    f.write("criterion      : {}\n".format(args.criterion))
    f.write("batch_size     : {}\n".format(args.batch_size))
    f.write("epochs         : {}\n".format(args.epochs))
    f.write("img_size       : {}\n".format(args.img_size))
    f.write("+++++++++++++++++++++++++++++++++++")
    f.close()
