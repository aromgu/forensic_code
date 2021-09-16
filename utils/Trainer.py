from time import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from tqdm import tqdm
import math
import tensorflow as tf
from utils import *
from utils.plot_functions import plot_test_results, plot_inputs

args = get_init()

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

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
        image, label = image.to(device), label.to(device)

        if args.patch_option == 'y':
            patched_image = get_patches(image, 3, args.patch_size)
            patched_label = get_patches(label, 1, args.patch_size)
            #
            # # RANDOM ERASE
            area = patched_image.size()[2] * patched_image.size()[3]
            r = np.random.rand(1)
            #
            if r < args.erase_prob:
                # for i in range(10):
                target_area = random.uniform(1e-5, 0.01) * area
                aspect_ratio = random.uniform(0.3, 1 / 0.3)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < patched_image.size()[3] and h < patched_image.size()[2]:
                    x1 = random.randint(0, patched_image.size()[2] - h)
                    y1 = random.randint(0, patched_image.size()[3] - w)

                    patched_image[:, :, x1:x1 + h, y1:y1 + w] = 0
                    patched_label[:, :, x1:x1 + h, y1:y1 + w] = 1

        # CUTOUT
        # lam = np.random.beta(args.beta, args.beta)
        # bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
        # image[:, :, bbx1:bbx2, bby1:bby2] = 0
        # label[:, :, bbx1:bbx2, bby1:bby2] = 1

        # CUTMIX
        # lam = np.random.beta(args.beta, args.beta)
        # rand_index = torch.randperm(image.size()[0]).to(device)
        # bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
        # image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
        # label[:, :, bbx1:bbx2, bby1:bby2] = 1
        #
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(image[0].permute(1,2,0).cpu().detach().numpy())
        # ax[1].imshow(label[0][0].cpu().detach().numpy(), cmap='gray')
        # ax[0].axis('off'); ax[1].axis('off')
        # plt.show()

            pred = model(patched_image)
            loss = criterion(pred, patched_label)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(img_patch[0].permute(1,2,0).cpu().detach().numpy())
        # ax[1].imshow(pred[0][0].cpu().detach().numpy(), cmap='gray')
        # ax[0].axis('off'); ax[1].axis('off')
        # plt.show()
        else : 
            pred = model(image)
            loss = criterion(pred, label)
        running_loss += loss.item()
        cnt += image.size(0)

        avg_loss = running_loss / cnt

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                running_loss / cnt))

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
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            pred = model(image)

            loss = criterion(pred, label)

            running_loss += loss.item()
            cnt += image.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(test_loader), np.round((batch_idx + 1) / len(test_loader) * 100.0, 2), avg_loss))

                # plot_inputs(image, resize(label), pred, pred, epoch, batch_idx+1)
                plot_test_results(image, resize(label), pred, epoch, batch_idx + 1)

            if (epoch == args.epochs or epoch == 1):

                acc, precision, recall, f1, iou = get_metrices(pred, label)

                acc_list.append(acc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)# .cpu().detach().numpy())
                recall_list.append(recall)#.cpu().detach().numpy())

    print("++++++++++ Test Report ++++++++++")
    print("mean Accuracy : ", np.mean(acc_list))
    print("mean Precision : ", np.mean(precision_list))
    print("mean Recall : ", np.mean(recall_list))
    print("mean F1 : ", np.mean(f1_list))
    print("mean IoU : ", np.mean(iou_list))
    print("++++++++++ Test Report ++++++++++")

    return avg_loss, np.mean(acc_list), np.mean(iou_list), np.mean(f1_list), np.mean(precision_list), np.mean(recall_list)

def fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, **kwargs):
    history = get_history()

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
