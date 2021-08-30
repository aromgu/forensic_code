import warnings
warnings.filterwarnings('ignore')

from torch import optim

from utils import *
from models import *
from Trainer import fit

def main(args):
    fix_seed(args.seed, device=args.device)

    train_loader, test_loader = load_dataloader(Tp_image_path=args.Tp_image_path,
                                                Tp_label_path=args.Tp_label_path,
                                                split_ratio=args.split_ratio,
                                                batch_size=args.batch_size,
                                                img_size=args.img_size)

    model = get_model(args.model_name)
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ',torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(args.device)

    criterion = nn.BCEWithLogitsLoss().to(args.device)
    # criterion = FocalLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    scheduler = get_scheduler(args, train_loader, optimizer)
    model, history = fit(scheduler, args.device, model, criterion, optimizer, train_loader, test_loader, args.epochs,
                         patch_module = get_patch_module,
                         net_loss_weight = args.net_loss_weight,
                         patch_loss_weight = args.patch_loss_weight,
                         get_patchnet=get_patch_conv(),
                         patch_option=args.patch_option,
                         get_fad = Fnet(args.img_size),
                         fad_option = args.fad_option,
                         model_name = args.model_name,
                         parent_dir = args.parent_dir)
    save_last_model(args.parent_dir, args.epochs, model, args.model_name, args.fad_option, args.patch_option) # iteration을 다 돌아서 나온 마지막 모델만 save 하는 것
    save_history(history, args.parent_dir, args.fad_option, args.patch_option, args.model_name, args.epochs)
    # history save

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