import matplotlib.pyplot as plt
from tqdm import tqdm
import kornia.color as k_color

import math
from utils.plot_functions import plot_test_results

from models.Canny import canny
from utils import *


args = get_init()

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

# fnet = Fnet(args.img_size).to(torch.device('cuda'))

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
    # loss1, loss2, edge_loss1, edge_loss2 = 0.0, 0.0, 0.0, 0.0

    for batch_idx, (image, label) in enumerate(train_loader):

        # start_time = time()
        low = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
        high = torch.zeros(image.size(0), 3, args.img_size, args.img_size)

        image, label = image.to(device), label.to(device)

        if args.patch_size != 0:
            image = get_patches(image, 3, args.patch_size)
            label = get_patches(label, 1, args.patch_size)

        # RANDOM ERASE
        area = image.size()[2] * image.size()[3]
        r = np.random.rand(1)
        if r < args.erase_prob:
            for i in range(10):
                target_area = random.uniform(1e-5, 0.01) * area
                aspect_ratio = random.uniform(0.3, 1 / 0.3)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                rand_index = torch.randperm(image.size()[0]).to(device)
                # print(rand_index)
                # for i in rand_index:
                #     if rand_index[i] == batch_idx:
                #         rand_index[i] = torch.randperm(1).to(device)
                # print(rand_index, 'after')
                if w < image.size()[3] and h < image.size()[2]:
                    x1 = random.randint(0, image.size()[2] - h)
                    y1 = random.randint(0, image.size()[3] - w)

                    image[:, :, x1:x1 + h, y1:y1 + w] = image[rand_index, :, x1:x1 + h, y1:y1 + w]
                    label[:, :, x1:x1 + h, y1:y1 + w] = 1

# target_area = random.uniform(1e-5, 0.01) * area
# aspect_ratio = random.uniform(0.3, 1 / 0.3)
#
# h = int(round(math.sqrt(target_area * aspect_ratio)))
# w = int(round(math.sqrt(target_area / aspect_ratio)))

# if w < image.size()[3] and h < image.size()[2]:
#     x1 = random.randint(0, image.size()[2] - h)
#     y1 = random.randint(0, image.size()[3] - w)
#
#     image[:, :, x1:x1 + h, y1:y1 + w] = 0
#     label[:, :, x1:x1 + h, y1:y1 + w] = 1             # Y, U, V = self._RGB2YUV(image) # apply transform only Y channel
#         fnet = Fnet(args.img_size)
#         fad_img = fnet(image)

        # 0.00034165382385253906s
        Y, U, V = _RGB2YUV(image)
        # 0.0011281967163085938s

        fad_img = kwargs['fnet'](Y)

        # 0.0042724609375s => 0.004785776138305664
        low[:,0,:,:] = fad_img[:,0,:,:]
        low[:,1,:,:] = U
        low[:,2,:,:] = V

        high[:,0,:,:] = fad_img[:,1,:,:]
        high[:,1,:,:] = U
        high[:,2,:,:] = V

        low = _YUV2RGB(low).to(torch.device('cuda'))
        high = _YUV2RGB(high).to(torch.device('cuda'))

        # plt.imshow(np.transpose(high[0].detach().cpu().numpy(), (1, 2, 0)))
        # plt.show()

        # 0.012476921081542969 => 0.006844043731689453
        canny_out = canny(label, device)
        canny_out[canny_out >= 0.5] = 1
        canny_out[canny_out < 0.5] = 0

        # region_model = UNetWithResnet50Encoder().to(torch.device('cuda'))
        # edge_model = UNetWithResnet50Encoder().to(torch.device('cuda'))

        # 0.3110980987548828s => 0.024767398834228516s => 0.022052288055419922
        # Server : 0.41074395179748535
        low_region_pred = model(low.float())

        high_edge_pred = model(high)

        # 0.6438965797424316s => 0.3725574016571045s => 0.10609579086303711 => 0.05857586860656738
        # Server : 2.416156053543091s
        # mid_edge_pred = edge_model(mid.cuda())

        high_edge_pred = torch.sigmoid(high_edge_pred)

        # mid_edge_pred = torch.sigmoid(mid_edge_pred)

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
        #     plt.imshow(image[0].permute(1, 2, 0).cpu().detach().numpy())
        #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        #     plt.tight_layout()
        #     plt.axis('off')
        #     plt.show()


        # plt.imshow(high[0].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        # plt.tight_layout()
        # plt.axis('off')
        # plt.show()

        # 0.7104315757751465s => 0.3872089385986328s => 0.06460356712341309s
        # Server : 2.3289411067962646
        loss1, bce, dice_loss = criterion(low_region_pred, label)
        # print(loss1)

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(canny_out[0].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
        # ax[0].axis('off')
        # ax[1].imshow(high[0].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
        # plt.tight_layout()
        # ax[1].axis('off')
        # plt.show()

        # 0.6506216526031494s => 0.3760662078857422s => 0.06825900077819824
        # Server : 2.2016425132751465s
        edge_loss, bce3, dice_loss3 = criterion(canny_out.float().cuda(), high_edge_pred)
        # edge_loss2 = hausdorff(canny_out, mid_edge_pred)
        #     if args.criterion == 'BCE' or  args.criterion == 'FL':
        #         loss = criterion(pred, label)
        #     else:
        #         loss, bce, dice_loss = criterion(pred, label)
        #
        # else :  # CUT MIX 조건
        #     pred = model(image)
        #
        #     if args.criterion == 'DICE': # 문자열만 있으면 항상 True >> args == '어쩌구' and args =='저쩌구' <<  필기 확인.
        #         loss, bce, dice_loss = criterion(pred, label)
        #     else:
        #         loss = criterion(pred, label)

        # 0.8229262828826904s => 0.464508056640625 => 0.37103748321533203 => 0.3516402244567871
        # Server : 2.749948263168335s
        loss = (loss1*0.5) + (edge_loss*0.5)

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

    for batch_idx, (image, label) in enumerate(test_loader):
        image, label = image.to(device), label.to(device)
        low = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
        high = torch.zeros(image.size(0), 3, args.img_size, args.img_size)

        with torch.no_grad():
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

            canny_out = canny(label, device)
            canny_out[canny_out >= 0.5] = 1
            canny_out[canny_out < 0.5] = 0

            # region = torch.cat((mid,low), dim=1)
            # region_model = UNetWithResnet50Encoder().to(torch.device('cuda'))
            # edge_model = UNetWithResnet50Encoder().to(torch.device('cuda'))
            low_region_pred = model(low.float())

            high_edge_pred = model(high)
            # mid_edge_pred = model(mid.cuda())

            high_edge_pred = torch.sigmoid(high_edge_pred)
            # mid_edge_pred = torch.sigmoid(mid_edge_pred)

            loss1, bce, dice_loss = criterion(low_region_pred, label)
            edge_loss, bce3, dice_loss3 = criterion(canny_out.float().cuda(), high_edge_pred)
            # edge_loss2 = hausdorff(canny_out, mid_edge_pred)

            # if args.criterion == 'BCE' or args.criterion == 'FL':
            #     loss = criterion(pred, label)
            # else:
            #     loss, bce, dice_loss = criterion(pred, label)

            loss = (loss1*0.5) + (edge_loss*0.5) #+ (edge_loss2*0.25)

            running_loss += loss.item()
            cnt += image.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
                print("++++++++++ Test Report ++++++++++")
                print("mean Accuracy : ", round(np.mean(acc_list), 4))
                print("mean Precision : ", round(np.mean(precision_list), 4))
                print("mean Recall : ", round(np.mean(recall_list), 4))
                print("mean F1 : ", round(np.mean(f1_list), 4))
                print("mean IoU : ", round(np.mean(iou_list), 4))
                print("++++++++++ Test Report ++++++++++")
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

            if (epoch == args.epochs or epoch > 0):
                pred = low_region_pred
                pred_np = F.sigmoid(pred.squeeze()).cpu().detach().numpy()
                label_np = label.squeeze().cpu().detach().numpy()

                pred_np[pred_np>=0.5]=1; pred_np[pred_np<0.5]=0 #

                acc, precision, recall, f1, iou = get_metrices(pred_np, label_np)

                acc_list.append(acc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)

            plot_test_results(image, resize(label), low_region_pred, high_edge_pred,  epoch, batch_idx + 1)

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
