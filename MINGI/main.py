import copy
import os
import warnings
# from utils.email import send_email
import torch
import optuna
import torch.nn.functional as F

warnings.filterwarnings('ignore')
from models import *
from models.model_core.FAD_LFS import Fnet
fit, test_epoch = get_trainer(args.trainer)

def main(args):
    device = get_device()
    fix_seed(device)

    train_loader, test_loader = get_dataloader(args)

    model = get_model(args.model_name, args, device)
    if args.load == 'y':
        load_path = os.path.join('res101_seg++', args.saved_pt)
        print("your model is loaded from {}".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(copy.deepcopy(checkpoint))

    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ',torch.cuda.device_count())
        model = nn.DataParallel(model)

    model.to(device)

    criterion = get_criterion(args).to(device)
    F_net = Fnet(args).to(device)

    ## OPTIMIZER ==================================
    # import adabound

    # pre_optim = adabound.AdaBound([param for name, param in model.named_parameters() if 'enc1' in name], lr=1e-5)
    # optimizer = adabound.AdaBound([param for name, param in model.named_parameters() if 'enc1' not in name], lr=1e-3)

    optimizer = get_optimizer(args, model)
    # optimizer = optim.Adam(list(model.parameters()) + list(F_net.parameters()), lr=args.learning_rate)
    # ==============================================================

    scheduler = get_scheduler(args, train_loader, optimizer)
    # pre_scheduler = get_scheduler(args, train_loader, pre_optim)
    if args.train :
        model, history, acc, iou, f1, precision, recall = fit(args, device, model, criterion, optimizer, scheduler, train_loader, test_loader,
                             fnet = F_net,
                             model_name = args.model_name,
                             parent_dir = args.parent_dir)
                             # optimizer2 = pre_optim,
                             # scheduler2 = pre_scheduler)
        save_last_model(args.parent_dir, args.epochs, model, optimizer, args.save_root)
        save_history(history, args.parent_dir, args.save_root)
        # send_email()
    else :
        # load_path = os.path.join(args.parent_dir,args.save_root,args.saved_pt)
        # load_path = os.path.join(args.save_root, args.saved_pt)
        # # model.load_state_dict(torch.load(load_path))

        # checkpoint = torch.load(load_path)
        # model.load_state_dict(checkpoint, strict=True)
        # for check_, model_ in zip(checkpoint.keys(), model.state_dict()) :
        #     if check_ != model_ :
        #         print("Shit! different key! : {} != {}".format(check_, model_))
        #     else :
        #         print("Shit! same key! : {} == {}".format(check_, model_))

        # print(checkpoint.keys())

        # model.load_state_dict(torch.load(load_path), strict=True)
        # model.to(device)


        # CHECK POINT INFERENCE ==
        # print("your model is loaded from {}".format(load_path))
        # checkpoint = torch.load(os.path.join(args.save_root,args.saved_pt), map_location=device)

        checkpoint = torch.load(os.path.join(args.parent_dir, args.save_root,args.saved_pt), map_location=device)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.to(device)
        print('Model to device')

        # model.load_state_dict(torch.load(load_path), strict=False)

    if args.trainer == 'autp':
        _, tp_auc, tp_iou, tp_f1, tp_precision, tp_recall, \
        au_auc, au_iou, au_f1, au_precision, au_recall = test_epoch(device, model, criterion, test_loader, args.epochs, fnet = F_net)
        save_metrics(args, args.parent_dir, args.save_root, [au_auc, au_iou, au_f1, au_precision, au_recall], save_path='au_result_metric.txt')
        save_metrics(args, args.parent_dir, args.save_root, [tp_auc, tp_iou, tp_f1, tp_precision, tp_recall], save_path='tp_result_metric.txt')
    else:
        print("INFERENCE")
        _, auc, iou, f1, precision, recall = test_epoch(args, device, model, criterion, test_loader, args.epochs, fnet=F_net)
        save_metrics(args, args.parent_dir, args.save_root, [auc, iou, f1, precision, recall], save_path='result_metric.txt')


def hyperparameter_tuning_main(trial) :
    cfg = {
        'learning_rate' : trial.suggest_categorical('learning_rate', [0.01, 0.001, 0.0001]),
        'batch_size' : trial.suggest_categorical('batch_size', [16, 32]),
        'criterion' : trial.suggest_categorical('criterion', ['DICE','FL']),
        'optimizer' : trial.suggest_categorical('optimizer', ['SGD','Adam','adabound']),
        'img_size' : trial.suggest_categorical('img_size', [256]),
        'diagonal' : trial.suggest_categorical('diagonal', [90, 100, 110, 120]),
    }
    args = get_init()
    args.learning_rate = cfg['learning_rate']
    args.batch_size = cfg['batch_size']
    args.criterion = cfg['criterion']
    args.optimizer = cfg['optimizer']
    args.img_size = cfg['img_size']
    args.diagonal = cfg['diagonal']

    # ===================

    device = get_device()
    fix_seed(device)

    train_loader, test_loader = get_dataloader(args)

    model = get_model(args.model_name, args, device)
    # parameter = list(model.named_parameters())

    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ',torch.cuda.device_count())
        model = nn.DataParallel(model)

    model.to(device)
    print('Model to device')

    criterion = get_criterion(args).to(device)
    F_net = Fnet(args).to(device)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, train_loader, optimizer)

    if args.train :

        model, history, acc, iou, f1, precision, recall = fit(args, device, model, criterion, optimizer, scheduler, train_loader, test_loader,
                             fnet = F_net,
                             model_name = args.model_name,
                             parent_dir = args.parent_dir)
                             # optimizer2 = pre_optim,
                             # scheduler2 = pre_scheduler)
        save_last_model(args.parent_dir, args.epochs, model, optimizer, args.save_root)
        save_history(history, args.parent_dir, args.save_root)
    else :
        model.load_state_dict(torch.load(os.path.join(args.parent_dir,args.save_root,args.saved_pt)), strict=False)

    _, auc, iou, f1, precision, recall = test_epoch(args, device, model, criterion, test_loader, args.epochs, fnet=F_net)
    save_metrics(args.parent_dir, args.save_root, [auc, iou, f1, precision, recall], save_path='result_metric.txt')

    del model
    return iou

if __name__ == '__main__':
    args = get_init()
    # ['dso1', 'coverage', 'casia1', 'casia2']
    for dataloader in ['dso1', 'coverage', 'casia1', 'casia2']:
        args.dataloader = dataloader
        args.save_root = "{}_{}".format(args.model_name, args.dataloader)
        main(args)

    # study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    # study.optimize(hyperparameter_tuning_main, n_trials=20)
    # print(study.best_params)
    # print(study.best_value)

    # for diagonal in [0,25,50,75,100,125,150,175,200,225,250,256] :
    #     args.diagonal = diagonal
    #     args.save_root = '18asppCBAM{}'.format(str(diagonal))


    # 파일 구조
    # Project File
    # |-- main.py
    # |-- Trainer.py -> train_epoch + test_epoch
    # |-- utils(Folder)
    #       |-- get_functions.py, save_funcions.py, print_functions.py, plot_functions.py
    # |-- models(Folder)
    #       |-- model.py
    #       |-- Our_Method(Folder)
    #           |-- gradient_extractor.py, ...

    # 원칙1 : 파일 및 폴더명 안겹치게 하기
    # 원칙2 : main.py 짧게 만들기
