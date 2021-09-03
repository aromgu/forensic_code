import warnings
warnings.filterwarnings('ignore')

from torch import optim

from utils import *
from models import *
from Trainer import fit

def main(args):
    device = get_device()
    fix_seed(device)

    train_loader, test_loader = load_dataloader(data_path=args.data_path,
                                                split_ratio=args.split_ratio,
                                                batch_size=args.batch_size,
                                                img_size=args.img_size)

    model = get_model(args.model_name, args, device)
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ',torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(device)

    criterion = get_criterion().to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, train_loader, optimizer)

    model, history = fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, args.epochs,
                         net_loss_weight = args.net_loss_weight,
                         low_loss_weight = args.low_loss_weight,
                         iou = iou_numpy,

                         mgp_option = args.mgp_option,
                         model_name = args.model_name,
                         parent_dir = args.parent_dir)
    save_last_model(args.parent_dir, args.epochs, model, args.model_name, args.fad_option)
    save_history(history, args.parent_dir, args.fad_option, args.model_name)

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