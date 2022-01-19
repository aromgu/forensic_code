# import matplotlib.pyplot as plt
from tqdm import tqdm
import kornia.color as k_color
import kornia

from utils.plot_functions import plot_test_results
from utils import *
from time import time
import torch.nn.functional as F

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

def train_epoch(args, device, model, criterion, optimizer, scheduler, train_loader, epoch, max_norm=1.0, **kwargs):
    """
    :param device:
    :param model:
    :param criterion:
    :param optimizer:
    :param train_loader:
    :param epoch:
    :param kwargs: (dictionary) patch 관련 성가셨던 요소들을 **kwargs 딕셔너리로 정리한 것.
    :return:
    """
    model.train()
    running_loss, cnt = 0.0, 0
    avg_loss = 0.0

    for batch_idx, (image, label) in tqdm(enumerate(train_loader)):
        image, label = image.to(device), label.to(device)

        edge_label = kornia.filters.sobel(label)
        edge_label[edge_label >= 0.3] = 1
        edge_label[edge_label < 0.3] = 0

        region_pred, edge_pred = model(image)

        region_loss = criterion(region_pred, label)
        edge_loss = criterion(edge_pred, edge_label)

        total_loss = (region_loss*0.5) + (edge_loss*0.5)

        running_loss += total_loss.item()
        cnt += image.size(0)

        optimizer.zero_grad()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):

            print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(args.model_name,
                epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                round(running_loss, 4) / cnt, format(total_loss,'.4f')))

    avg_loss = running_loss / cnt

    return avg_loss

def test_epoch(args, device, model, criterion, test_loader, epoch, **kwargs) :
    model.eval()
    running_loss, cnt = 0.0, 0
    avg_loss = 0.0
    auc_list = []
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    for batch_idx, (image, label) in enumerate(test_loader):
        with torch.no_grad():
            image, label = image.to(device), label.to(device)

            edge_label = kornia.filters.sobel(label)
            edge_label[edge_label >= 0.3] = 1
            edge_label[edge_label < 0.3] = 0

            region_pred, edge_pred = model(image)

            region_loss = criterion(region_pred, label)
            edge_loss = criterion(edge_pred, edge_label)

            total_loss = (region_loss * 0.5) + (edge_loss * 0.5)

            running_loss += total_loss.item()
            cnt += image.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    args.model_name,
                    epoch, batch_idx + 1, len(test_loader),
                    np.round((batch_idx + 1) / len(test_loader) * 100.0, 2),
                    round(running_loss, 4) / cnt, format(total_loss,'.4f')))

                plot_test_results(image, resize(label), region_pred, epoch, batch_idx + 1, [0.0,0.0,0.0,0.0], save_root_path=args.save_root)

            if epoch == args.epochs or epoch == 1:
                pred = F.sigmoid(region_pred)

                auc, precision, recall, f1, iou = get_metrices(label, pred)

                if precision is np.isnan(precision) :
                    sys.exit(1)
                auc_list.append(auc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)

            if epoch == args.epochs:
                plot_test_results(image, resize(label), pred, epoch, batch_idx + 1, [f1,iou,precision,recall], save_root_path=args.save_root)

    avg_loss = running_loss / cnt  # 무야호 춧

    return round(avg_loss,4), round(np.mean(auc_list),4), round(np.mean(iou_list),4), \
           round(np.mean(f1_list),4), round(np.mean(precision_list),4), round(np.mean(recall_list),4)

def fit(args, device, model, criterion, optimizer, scheduler, train_loader, test_loader, **kwargs):
    history = get_history()
    print_configurations(args)
    auc, iou, f1, precision, recall = 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch in tqdm(range(1, args.epochs + 1)) :
        start_time = time()

        print("TRAINING")
        train_loss = train_epoch(args, device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs)

        print("EVALUATE")
        test_loss, auc, iou, f1, precision, recall = test_epoch(args, device, model, criterion, test_loader, epoch, **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}| current_lr:{get_current_lr(optimizer)} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history, auc, iou, f1, precision, recall
