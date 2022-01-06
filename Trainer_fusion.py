# import matplotlib.pyplot as plt
from tqdm import tqdm
import kornia.color as k_color

from utils.plot_functions import plot_test_results
# import torch.nn.functional as fuck
from utils import *
# import math
# import kornia
# from models.l1pruner import filter_l1_pruning
BCE = nn.BCEWithLogitsLoss()

args = get_init()
# from utils.plot_functions import plot_feature_maps

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

def _RGB2YUV(img):
    YUV = k_color.rgb_to_yuv(img)

    return YUV[..., 0, :, :].unsqueeze(dim=1), YUV[..., 1, :, :], YUV[..., 2, :, :]

def _YUV2RGB(img):
    RGB = k_color.yuv_to_rgb(img)

    return RGB

# MSE = nn.MSELoss()

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

        # start_time = time()
        # low = torch.zeros_like(image)
        # high = torch.zeros_like(image)

        image, label = image.to(device), label.to(device)

        # if args.patch_size != 0:
        #     image = get_patches(image, 3, args.patch_size)
        #     label = get_patches(label, 1, args.patch_size)
        # ===== CUTMIX ============
        # area = image.size()[2] * image.size()[3]
        # r = np.random.rand(1)
        # if r < args.erase_prob:
        #     for i in range(10):
        #         target_area = random.uniform(1e-5, 0.01) * area
        #         aspect_ratio = random.uniform(0.3, 1 / 0.3)
        #
        #         h = int(round(math.sqrt(target_area * aspect_ratio)))
        #         w = int(round(math.sqrt(target_area / aspect_ratio)))
        #         rand_index = torch.randperm(image.size()[0]).to(device)
        #         if w < image.size()[3] and h < image.size()[2]:
        #             x1 = random.randint(0, image.size()[2] - h)
        #             y1 = random.randint(0, image.size()[3] - w)
        #
        #             image[:, :, x1:x1 + h, y1:y1 + w] = image[rand_index, :, x1:x1 + h, y1:y1 + w]
        #             label[:, :, x1:x1 + h, y1:y1 + w] = 1
        # ===== CUTMIX ============
        # Y, U, V = _RGB2YUV(image)

        fad_img = kwargs['fnet'](image)

        # low[:,0,:,:] = fad_img[:,0,:,:]
        # low[:,1,:,:] = U
        # low[:,2,:,:] = V
        #
        # high[:,0,:,:] = fad_img[:,1,:,:]
        # high[:,1,:,:] = U
        # high[:,2,:,:] = V

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(high[0].permute(1,2,0).cpu().detach().numpy(), cmap='gray')
        # ax[1].imshow(low[0].permute(1,2,0).cpu().detach().numpy())
        # plt.show()

        # low = _YUV2RGB(low).to(device)
        # high = _YUV2RGB(high).to(device)

        # kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        # edge_image = kornia.morphology.erosion(image, kernel.cuda())
        # edge_image = torch.mean(torch.sub(image, edge_image), dim=1)

        # edge_label = kornia.filters.sobel(label)

        high = fad_img[:, 3:, :, :]
        low = fad_img[:, :3, :, :]
        prediction = model(image, low, high)
        # prediction = model(image)
        if args.criterion == 'DICE':
            loss, bce, dice_loss = criterion(prediction, label)
        else:
            loss = criterion(prediction, label)
        # bce_loss = BCE(edge_pred, edge_label.cuda())
        # unet_loss = MSE(unet_out, edge_image.cuda())
        # total_loss = (loss + bce_loss) / 2

        running_loss += loss.item()
        cnt += image.size(0)

        avg_loss = running_loss / cnt
        optimizer.zero_grad()
        # end_time = time() - start_time
        # print("before backward : ", end_time)
        loss.backward()
        # end_time = time() - start_time
        # print("after backward : ", end_time)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
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
    auc_list = []
    iou_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    for batch_idx, (image, label) in enumerate(test_loader):
        with torch.no_grad():
            image, label = image.to(device), label.to(device)
            # low = torch.zeros_like(image)  # low = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
            # high = torch.zeros_like(image)  # high = torch.zeros(image.size(0), 3, args.img_size, args.img_size)
            # Y, U, V = _RGB2YUV(image)
            fad_img = kwargs['fnet'](image)

            # low[:, 0, :, :] = fad_img[:, 0, :, :]
            # low[:, 1, :, :] = U
            # low[:, 2, :, :] = V
            #
            # high[:, 0, :, :] = fad_img[:, 1, :, :]
            # high[:, 1, :, :] = U
            # high[:, 2, :, :] = V

            # low = _YUV2RGB(low).to(torch.device('cuda'))
            # high = _YUV2RGB(high).to(torch.device('cuda'))

            # kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            # edge_image = kornia.morphology.erosion(image, kernel.cuda())
            # edge_image = torch.mean(torch.sub(image, edge_image), dim=1)

            # edge_label = kornia.filters.sobel(label)
            low = fad_img[:,:3,:,:]
            high = fad_img[:,3:,:,:]
            # prediction, out1, out2, out3 = model(image, low, high)
            prediction = model(image, low, high)

            if args.criterion == 'DICE':
                loss, bce, dice_loss = criterion(prediction, label)
            else:
                loss = criterion(prediction, label)
            # bce_loss = BCE(edge_pred, edge_label.cuda())

            # if epoch==args.epochs and torch.sum(label)==0:
            #     pass
            # else:
            #     plot_feature_maps('refine34_seam_', image, prediction, label, out1, out2, out3, epoch, batch_idx) ### FEATURE MAPS


            # total_loss = (loss+bce_loss) / 2

            running_loss += loss.item()
            cnt += image.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 400 == 0 or (batch_idx + 1) == len(test_loader):
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


            if epoch == args.epochs: # if (epoch == args.epoch) or epoch > 0 :
                pred = prediction
                pred_np = F.sigmoid(pred.squeeze()).cpu().detach().numpy()
                pred_ = F.sigmoid(pred.squeeze()).cpu().detach().numpy()
                label_np = label.squeeze().cpu().detach().numpy()

                pred_np[pred_np>=0.5]=1; pred_np[pred_np<0.5]=0

                auc, precision, recall, f1, iou = get_metrices(pred_np, label_np, pred_)
                if precision is np.isnan(precision) :
                    sys.exit(1)
                auc_list.append(auc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)

                ## change
                plot_test_results(image, resize(label), prediction, epoch, batch_idx + 1)

    return round(avg_loss,4), round(np.mean(auc_list),4), round(np.mean(iou_list),4), \
           round(np.mean(f1_list),4), round(np.mean(precision_list),4), round(np.mean(recall_list),4)

def fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, **kwargs):
    history = get_history()
    auc, iou, f1, precision, recall = 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch in tqdm(range(1, epochs + 1)) :
        start_time = time()

        print("TRAINING")
        train_loss = train_epoch(device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs)

        print("EVALUATE")
        test_loss, auc, iou, f1, precision, recall = test_epoch(device, model, criterion, test_loader, epoch, **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}| current_lr:{get_current_lr(optimizer)} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history, auc, iou, f1, precision, recall
