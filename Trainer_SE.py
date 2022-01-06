import matplotlib.pyplot as plt
from tqdm import tqdm
# import kornia.losses.FocalLoss
from utils.plot_functions import plot_test_results

from utils import *
import torch.nn.functional as F
import kornia
args = get_init()

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)
MSE = nn.MSELoss()
BCE = nn.BCEWithLogitsLoss()

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
        # edge_label = canny(label, device)

        # kernel = np.ones((5, 5), np.uint8)
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        edge_image = kornia.morphology.erosion(image, kernel.cuda())
        edge_image = torch.mean(torch.sub(image,edge_image), dim=1)

        edge_label = kornia.filters.sobel(label)

        img_edges, mask_edges, label_mask = model(image)
        img_edges = F.sigmoid(img_edges)

        mse_loss = MSE(img_edges, edge_image.cuda())
        bce_loss = BCE(mask_edges, edge_label.cuda())
        # fl_loss = focal(label_mask, label, alpha=0.25)
        fl_loss = criterion(label_mask, label)

        total_loss = (mse_loss + bce_loss + fl_loss) / 3

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
                print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | Running Loss : {} | FL : {} | MSE : {} | BCE : {}".format(
                    args.model_name,
                    epoch, batch_idx + 1, len(train_loader),
                    np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                           round(running_loss, 4) / cnt, fl_loss, mse_loss, bce_loss))

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
            # edge_label = canny(label, device)

            kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            edge_image = kornia.morphology.erosion(image, kernel.cuda())
            edge_image = torch.mean(torch.sub(image, edge_image), dim=1)

            edge_label = kornia.filters.sobel(label)

            img_edges, mask_edges, label_mask = model(image)
            img_edges = F.sigmoid(img_edges)

            mse_loss = MSE(img_edges, edge_image.cuda())
            bce_loss = BCE(mask_edges, edge_label.cuda())
            # fl_loss = focal(label_mask, label, alpha=0.25)
            fl_loss = criterion(label_mask, label)

            total_loss = (mse_loss + bce_loss + fl_loss) / 3

            fig, ax = plt.subplots(1,4)
            ax[0].imshow(image[0].permute(1,2,0).cpu().detach().cpu())
            ax[1].imshow(label[0].permute(1,2,0).cpu().detach().cpu(), cmap='gray')
            ax[2].imshow(mask_edges[0].permute(1,2,0).cpu().detach().cpu(), cmap='gray')
            ax[3].imshow(label_mask[0].permute(1,2,0).cpu().detach().cpu(), cmap='gray')
            plt.show()

            running_loss += total_loss.item()
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
                    print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | Running Loss : {} | FL : {} | MSE : {} | BCE : {}".format(
                        args.model_name,
                        epoch, batch_idx + 1, len(test_loader),
                        np.round((batch_idx + 1) / len(test_loader) * 100.0, 2),
                        round(running_loss, 4) / cnt, fl_loss, mse_loss, bce_loss))

                plot_test_results(image, resize(label), label_mask, epoch, batch_idx + 1)


            if epoch == args.epochs: # if (epoch == args.epoch) or epoch > 0 :
                pred = label_mask
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
                plot_test_results(image, resize(label), label_mask, epoch, batch_idx + 1)

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
