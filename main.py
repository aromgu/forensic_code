import os
import warnings
warnings.filterwarnings('ignore')

from models import *
fit, test_epoch = get_trainer(args.trainer)

def main(args):
    device = get_device()
    fix_seed(device)
# load_tpau
    train_loader, test_loader = get_dataloader(args)

    model = get_model(args.model_name, args, device)
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ',torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.to(device)
    print('Model to device')

    criterion = get_criterion(args).to(device)
    F_net = Fnet(args.img_size, diagonal=args.diagonal).to(device)
    # optimizer = optim.Adam(list(model.parameters()) + list(F_net.parameters()), lr=args.learning_rate)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, train_loader, optimizer)
    if args.train :

        model, history, acc, iou, f1, precision, recall = fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, args.epochs,
                             fnet = F_net,
                             model_name = args.model_name,
                             parent_dir = args.parent_dir)
        save_last_model(args.parent_dir, args.epochs, model, optimizer, args.save_root)
        save_history(history, args.parent_dir, args.save_root)
    else :
        # model = model.load_state_dict(torch.load(os.path.join(args.parent_dir,args.save_root,args.saved_pt)))
        # print('=== Inference Path : ',os.path.join(args.parent_dir,args.save_root,args.saved_pt),' ===')
        model.load_state_dict(torch.load(os.path.join(args.parent_dir,args.save_root,args.saved_pt)))
        # print(torch.load(os.path.join('MantraNetv4.pt')).keys())
        # model.load_state_dict(torch.load(os.path.join('MantraNetv4.pt')))
        _, acc, iou, f1, precision, recall = test_epoch(device, model, criterion, test_loader, 100, fnet = F_net)
    save_metrics(args.parent_dir, args.save_root, [acc, iou, f1, precision, recall], save_path='result_metric.txt')

if __name__ == '__main__':
    args = get_init()
    for diagonal in [175,200,220,250,256] :
       args.diagonal = diagonal
       args.save_root = '18ASPPfilter{}'.format(str(diagonal))



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
