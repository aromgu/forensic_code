import os

import torch

from utils.plot_functions import plot_loss

def save_last_model(parent_dir, epoch, model, optimizer, save_root):
    PATH = os.path.join(parent_dir,save_root)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(PATH, "last_{}.pth".format(epoch)))
        print(f'saved last model : epoch{epoch}')
    else:
        torch.save(model.state_dict(), os.path.join(PATH, "last_{}.pth".format(epoch)))
        print(f'saved last model : epoch{epoch}')

        # check_point = {
        #     'model': model.module if torch.cuda.device_count() > 1 else model,
        #     'model_name': model,
        #     'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model,
        #     'optimizer_state_dict': optimizer.state_dict()
        # }
        #
        # torch.save(check_point,
        #            os.path.join(PATH, '{}.pth'.format(epoch)))

def save_history(history, parent_dir, save_root):
    plot_loss(parent_dir, history, save_root)
    f = open(os.path.join(parent_dir, save_root)+'/history.txt', 'w')
    f.write(str(history))
    f.close()

def save_metrics(parent_dir, save_root, metrcis, save_path) :
    print("+++++++++++ TEST REPORT +++++++++++")
    print("AUC : {}\n".format(metrcis[0]))
    print("IoU : {}\n".format(metrcis[1]))
    print("F1-score : {}\n".format(metrcis[2]))
    print("Precision : {}\n".format(metrcis[3]))
    print("Recall : {}\n".format(metrcis[4]))
    print("+++++++++++ TEST REPORT +++++++++++")

    f = open(os.path.join(parent_dir, save_root, save_path), 'w')

    f.write("###################### TEST REPORT ######################\n")
    f.write("AUC : {}\n".format(metrcis[0]))
    f.write("IoU : {}\n".format(metrcis[1]))
    f.write("F1-score : {}\n".format(metrcis[2]))
    f.write("Precision : {}\n".format(metrcis[3]))
    f.write("Recall : {}\n".format(metrcis[4]))
    f.write("###################### TEST REPORT ######################")

    f.close()