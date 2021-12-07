from tqdm import tqdm
import kornia.color as k_color

from utils.plot_functions import plot_test_results

from utils import *

args = get_init()

focal = FocalLoss()

resize = transforms.Compose(
    [transforms.Resize((args.img_size, args.img_size))]
)

def _RGB2YUV(img):
    YUV = k_color.rgb_to_yuv(img)

    return YUV[..., 0, :, :].unsqueeze(dim=1), YUV[..., 1, :, :], YUV[..., 2, :, :]

def _YUV2RGB(img):
    RGB = k_color.yuv_to_rgb(img)

    return RGB

# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int32(W * cut_rat)
#     cut_h = np.int32(H * cut_rat)
#
#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
#
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#
#     return bbx1, bby1, bbx2, bby2


def train_epoch(device, model, criterion, optimizer, scheduler, train_loader, epoch, max_norm=1.0, **kwargs):
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

    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        o1, o2, o3, o4 = model(image)
        loss1, bce, dice_loss = criterion(o1, label)
        loss2, bce, dice_loss = criterion(o2, label)
        loss3, bce, dice_loss = criterion(o3, label)
        loss4, bce, dice_loss = criterion(o4, label)

        focal1 = focal(o1, label)
        focal2 = focal(o2, label)
        focal3 = focal(o3, label)
        focal4 = focal(o4, label)

        total_loss = (loss1 + loss2 + loss3 + loss4 + focal1 + focal2 + focal3 + focal4) / 8
        running_loss += total_loss.item()
        cnt += image.size(0)

        avg_loss = running_loss / cnt

        optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            if args.criterion == 'DICE':
                print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}(bce : {} | dice : {})".format(args.model_name,
                     epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                   round(running_loss,4) / cnt, format(bce,'.4f'), format(dice_loss,'.4f')))
            else:
                print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(args.model_name,
                    epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                    round(running_loss, 4) / cnt, format(total_loss,'.4f')))

    return avg_loss

def test_epoch(device, model, criterion, test_loader, epoch, **kwargs) :
    model.eval()
    running_loss, cnt = 0.0, 0
    avg_loss = 0.0
    acc_list = []
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    for batch_idx, (image, label) in enumerate(test_loader):
        with torch.no_grad():
            image, label = image.to(device), label.to(device)

            o1, o2, o3, o4 = model(image)
            loss1, bce, dice_loss = criterion(o1, label)
            loss2, bce, dice_loss = criterion(o2, label)
            loss3, bce, dice_loss = criterion(o3, label)
            loss4, bce, dice_loss = criterion(o4, label)

            focal1 = focal(o1, label)
            focal2 = focal(o2, label)
            focal3 = focal(o3, label)
            focal4 = focal(o4, label)

            total_loss = (loss1+loss2+loss3+loss4 + focal1+focal2+focal3+focal4) / 8
            running_loss += total_loss.item()
            cnt += image.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == len(test_loader):
                if args.criterion == 'DICE':
                    print(
                        "{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}(bce : {} | dice : {})".format(
                            args.model_name,
                            epoch, batch_idx + 1, len(test_loader),
                            np.round((batch_idx + 1) / len(test_loader) * 100.0, 2),
                            round(running_loss, 4) / cnt, format(bce,'.4f'), format(dice_loss,'.4f')))
                else:
                    print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                        args.model_name,
                        epoch, batch_idx + 1, len(test_loader),
                        np.round((batch_idx + 1) / len(test_loader) * 100.0, 2),
                        round(running_loss, 4) / cnt, format(total_loss,'.4f')))

                plot_test_results(image, resize(label), o4, epoch, batch_idx + 1)


            if epoch == args.epochs:
                pred = o4
                pred_np = F.sigmoid(pred.squeeze()).cpu().detach().numpy()
                label_np = label.squeeze().cpu().detach().numpy()

                pred_np[pred_np>=0.5]=1; pred_np[pred_np<0.5]=0

                acc, precision, recall, f1, iou = get_metrices(pred_np, label_np)
                # print(precision)
                if precision is np.isnan(precision) :
                    sys.exit(1)
                acc_list.append(acc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)

                ## change
                plot_test_results(image, resize(label), o4, epoch, batch_idx + 1)

    return round(avg_loss,4), round(np.mean(acc_list),4), round(np.mean(iou_list),4), \
           round(np.mean(f1_list),4), round(np.mean(precision_list),4), round(np.mean(recall_list),4)

def fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, **kwargs):
    history = get_history()
    acc, iou, f1, precision, recall = 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch in tqdm(range(1, epochs + 1)) :
        start_time = time()

        print("TRAINING")
        train_loss = train_epoch(device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs)

        print("EVALUATE")
        test_loss, acc, iou, f1, precision, recall = test_epoch(device, model, criterion, test_loader, epoch, **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}| current_lr:{get_current_lr(optimizer)} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history, acc, iou, f1, precision, recall
