import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import kornia.color as k_color

import math
from utils.plot_functions import plot_test_results

from models.Canny import canny
from utils import *
from collections import OrderedDict
import pickle

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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

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

        # SPADE
        # set model's intermediate outputs
        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)
        model.avgpool.register_forward_hook(hook)
        os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_img_rocauc = ax[0]
        fig_pixel_rocauc = ax[1]

        total_roc_auc = []
        total_pixel_roc_auc = []
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])


        # start_time = time()
        low = torch.zeros_like(image)
        high = torch.zeros_like(image)

        image, label = image.to(device), label.to(device)

        if args.patch_size != 0:
            image = get_patches(image, 3, args.patch_size)
            label = get_patches(label, 1, args.patch_size)
        # end_time = time() - start_time
        # print("time : ", end_time)
        # 0.017838239669799805
        Y, U, V = _RGB2YUV(image)
        # 0.017165184020996094

        fad_img = kwargs['fnet'](Y)
        # 0.02581930160522461
        low[:,0,:,:] = fad_img[:,0,:,:]
        low[:,1,:,:] = U
        low[:,2,:,:] = V

        high[:,0,:,:] = fad_img[:,1,:,:]
        high[:,1,:,:] = U
        high[:,2,:,:] = V

        low = _YUV2RGB(low).to(torch.device('cuda'))
        high = _YUV2RGB(high).to(torch.device('cuda'))

        # fig, ax = plt.subplots(1,4)
        # ax[0].imshow(image[0].permute(1, 2, 0).cpu().detach().numpy())
        # ax[1].imshow(low[0].permute(1, 2, 0).cpu().detach().numpy())
        # ax[2].imshow(high[0].permute(1, 2, 0).cpu().detach().numpy())
        # ax[3].imshow(label[0].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
        # plt.show()

        # 0.0522160530090332
        prediction = model(image, low.float(), high)
        # prediction = model(image)
        # 0.12936878204345703

        # SPADE
        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, y, mask) in tqdm(train_loader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    pred = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v)
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)
            # save extracted feature
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)


        loss, bce, dice_loss = criterion(prediction, label)
        # 0.12885332107543945

        running_loss += loss.item()
        cnt += image.size(0)

        avg_loss = running_loss / cnt

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            if args.criterion == 'DICE':
                print("{} patch size {}| Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}(bce : {} | dice : {})".format(args.model_name,
                    args.patch_size, epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                   round(running_loss,4) / cnt, format(bce,'.4f'), format(dice_loss,'.4f')))
            else:
                print("{} patch size {}| Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(args.model_name,
                    args.patch_size, epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
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
    gt_list = []
    gt_mask_list = []
    test_imgs = []

    for batch_idx, (image, label) in enumerate(test_loader):
        with torch.no_grad():
            image, label = image.to(device), label.to(device)

            # change
            low = torch.zeros_like(image)  # low = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
            high = torch.zeros_like(image)  # high = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
            Y, U, V = _RGB2YUV(image)

            fad_img = kwargs['fnet'](Y)

            low[:, 0, :, :] = fad_img[:, 0, :, :]
            low[:, 1, :, :] = U
            low[:, 2, :, :] = V

            high[:, 0, :, :] = fad_img[:, 1, :, :]
            high[:, 1, :, :] = U
            high[:, 2, :, :] = V

            low = _YUV2RGB(low).to(torch.device('cuda'))
            high = _YUV2RGB(high).to(torch.device('cuda'))

            prediction = model(image, low.float(), high)
            # prediction = model(image)

            loss, bce, dice_loss = criterion(prediction, label)

            running_loss += loss.item()
            cnt += image.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                if args.criterion == 'DICE':
                    print(
                        "{} patch size {}| Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}(bce : {} | dice : {})".format(
                            args.model_name,
                            args.patch_size, epoch, batch_idx + 1, len(test_loader),
                            np.round((batch_idx + 1) / len(test_loader) * 100.0, 2),
                            round(running_loss, 4) / cnt, format(bce,'.4f'), format(dice_loss,'.4f')))
                else:
                    print("{} patch size {}| Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                        args.model_name,
                        args.patch_size, epoch, batch_idx + 1, len(test_loader),
                        np.round((batch_idx + 1) / len(test_loader) * 100.0, 2),
                        round(running_loss, 4) / cnt, format(loss,'.4f')))
                ## change
                plot_test_results(image, resize(label), prediction, epoch, batch_idx + 1)

            ## change
            if epoch == args.epochs: # if (epoch == args.epoch) or epoch > 0 : => always?...
                pred = prediction
                pred_np = F.sigmoid(pred.squeeze()).cpu().detach().numpy()
                label_np = label.squeeze().cpu().detach().numpy()

                pred_np[pred_np>=0.5]=1; pred_np[pred_np<0.5]=0

                acc, precision, recall, f1, iou = get_metrices(pred_np, label_np)

                acc_list.append(acc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)

                ## change
                plot_test_results(image, resize(label), prediction, epoch, batch_idx + 1)

    print("++++++++++ Test Report ++++++++++")
    print("mean Accuracy : ", round(np.mean(acc_list),4))
    print("mean Precision : ", round(np.mean(precision_list),4))
    print("mean Recall : ", round(np.mean(recall_list),4))
    print("mean F1 : ", round(np.mean(f1_list),4))
    print("mean IoU : ", round(np.mean(iou_list),4))
    print("++++++++++ Test Report ++++++++++")

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
