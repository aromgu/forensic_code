from tqdm import tqdm
import kornia.color as k_color

from utils.plot_functions import plot_test_results

from utils import *

args = get_init()

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

def _RGB2YUV(img):
    YUV = k_color.rgb_to_yuv(img)

    return YUV[..., 0, :, :].unsqueeze(dim=1), YUV[..., 1, :, :], YUV[..., 2, :, :]

def _YUV2RGB(img):
    RGB = k_color.yuv_to_rgb(img)

    return RGB

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

        start_time = time()
        low = torch.zeros_like(image)
        high = torch.zeros_like(image)

        image, label = image.to(device), label.to(device)

        Y, U, V = _RGB2YUV(image)

        fad_img = kwargs['fnet'](Y)

        low[:,0,:,:] = fad_img[:,0,:,:]
        low[:,1,:,:] = U
        low[:,2,:,:] = V

        high[:,0,:,:] = fad_img[:,1,:,:]
        high[:,1,:,:] = U
        high[:,2,:,:] = V

        low = _YUV2RGB(low).to(device)
        high = _YUV2RGB(high).to(device)

        # prediction = model(image)
        prediction = model(image, low, high)

        loss, bce, dice_loss = criterion(prediction, label)
        # loss = criterion(prediction, label)

        running_loss += loss.item()
        cnt += image.size(0)

        avg_loss = running_loss / cnt

        optimizer.zero_grad()

        # end_time = time() - start_time
        # print("before backward time : ", end_time)

        loss.backward()

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
                    round(running_loss, 4) / cnt, format(loss,'.4f')))

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

            # change
            low = torch.zeros_like(image)  # low = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
            high = torch.zeros_like(image)  # high = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
            Y, U, V = _RGB2YUV(image)
            #
            fad_img = kwargs['fnet'](Y)

            low[:, 0, :, :] = fad_img[:, 0, :, :]
            low[:, 1, :, :] = U
            low[:, 2, :, :] = V

            high[:, 0, :, :] = fad_img[:, 1, :, :]
            high[:, 1, :, :] = U
            high[:, 2, :, :] = V

            low = _YUV2RGB(low).to(torch.device('cuda'))
            high = _YUV2RGB(high).to(torch.device('cuda'))

            prediction = model(image, low, high)

            loss, bce, dice_loss = criterion(prediction, label)

            running_loss += loss.item()
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
                        round(running_loss, 4) / cnt, format(loss,'.4f')))

                plot_test_results(image, resize(label), prediction, epoch, batch_idx + 1)

            if epoch == args.epochs:
                pred = prediction
                pred_np = F.sigmoid(pred.squeeze()).cpu().detach().numpy()
                label_np = label.squeeze().cpu().detach().numpy()

                pred_np[pred_np>=0.5]=1; pred_np[pred_np<0.5]=0

                acc, precision, recall, f1, iou = get_metrices(pred_np, label_np)
                if precision is np.isnan(precision) :
                    sys.exit(1)
                acc_list.append(acc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)

                ## change
                plot_test_results(image, resize(label), prediction, epoch, batch_idx + 1)

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
