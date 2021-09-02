# model을 하나로 integration을 시켜야할듯.
# 모델이 너무 scatter 되있어서 보기 어려움
# 하나의 모듈 클래스를 만들어서 거기에 다 넣는게 좋을듯
    # model = MGP + canny edge detection + U-net
# 필요없는 argument 없애기
    # fit : get patch? get_fad? fad_option?
# 사실 상 best 모델은 굳이?... 저장할 필요 없는 거 같은데, 어차피 마지막 에폭 모델만 비교해야됨

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from Trainer import fit

def main(args):
    fix_seed(args.seed, device=args.device)
    train_loader, test_loader = load_dataloader(data_path=args.data_path,
                                                split_ratio=args.split_ratio,
                                                batch_size=args.batch_size,
                                                img_size=args.img_size)

    model = get_model(args.model_name)
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ',torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(args.device)
    model.apply(weights_init)
    print("Model weight initialization complete!")

    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, train_loader, optimizer)

    model, history = fit(scheduler, args.device, model, criterion, optimizer, train_loader, test_loader, args.epochs,
                         MGP = extract,
                         net_loss_weight = args.net_loss_weight,
                         low_loss_weight = args.low_loss_weight,
                         get_patchnet=get_patch_conv(),
                         get_fad=Fnet(args.img_size),
                         low_freq = lowfreq_mask,
                         iou = iou_numpy,
                         canny = canny,

                         fad_option = args.fad_option,
                         mgp_option = args.mgp_option,
                         model_name = args.model_name,
                         parent_dir = args.parent_dir)
    save_last_model(args.parent_dir, args.epochs, model, args.model_name, args.fad_option) # iteration을 다 돌아서 나온 마지막 모델만 save 하는 것
    save_history(history, args.parent_dir, args.fad_option, args.model_name)
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