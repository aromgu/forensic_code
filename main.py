import os
import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from Trainer import fit, test_epoch

def main(args):
    device = get_device()
    fix_seed(device)

    train_loader, test_loader = load_dataloader(data_path=args.data_path,
                                                split_ratio=args.split_ratio,
                                                batch_size=args.batch_size,
                                                img_size=args.img_size,
                                                num_workers=args.num_workers)

    model = get_model(args.model_name, args, device)
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ',torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(device)

    criterion = get_criterion(args).to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, train_loader, optimizer)

    if args.train :
        model, history, acc, iou, f1, precision, recall = fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, args.epochs,
                             high_loss_weight = args.high_loss_weight,
                             low_loss_weight = args.low_loss_weight,
                             model_name = args.model_name,
                             parent_dir = args.parent_dir)
        save_last_model(args.parent_dir, args.epochs, model, optimizer, args.save_root)
        save_history(history, args.parent_dir, args.save_root)
    else :
        # model = model.load_state_dict(torch.load(os.path.join(args.parent_dir,args.save_root,args.saved_pt)))
        model.load_state_dict(torch.load(os.path.join(args.parent_dir,args.save_root,args.saved_pt)))
        _, acc, iou, f1, precision, recall = test_epoch(device, model, criterion, test_loader, 100)
    save_metrics(args.parent_dir, args.save_root, [acc, iou, f1, precision, recall], save_path='result_metric.txt')

if __name__ == '__main__':
    args = get_init()
    main(args)

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