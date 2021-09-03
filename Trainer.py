from time import time

from tqdm import tqdm

import configuration
from utils.plot_functions import plot_test_results, plot_mgp

from utils import *
args = get_init()

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

def train_epoch(device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs):
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

        high_freq_output, low_freq_output, edge_GT = model(image, label, device)

        optimizer.zero_grad()

        # LOSS
        high_loss = criterion(high_freq_output.to(device), edge_GT.to(device))
        low_loss = criterion(low_freq_output.to(device), label)

        total_loss = kwargs['net_loss_weight'] * high_loss + kwargs['low_loss_weight'] * low_loss

        running_loss += total_loss.item()
        cnt += image.size(0)

        avg_loss = running_loss / cnt

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                running_loss / cnt
            ))

    return avg_loss

def test_epoch(device, model, criterion, optimizer, test_loader, epoch, **kwargs) :
    model.eval()
    running_loss, cnt = 0.0, 0
    avg_loss = 0.0
    for batch_idx, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            optimizer.zero_grad()

            mgp_output, edge_GT, lowfreq, spectrum, mask_list, k = model(X, y, device)

            optimizer.zero_grad()

            # LOSS
            high_loss = criterion(mgp_output.to(device), edge_GT.to(device))
            low_loss = criterion(lowfreq.to(device), y)

            total_loss = kwargs['net_loss_weight'] * high_loss + kwargs['low_loss_weight'] * low_loss

            high_iou = kwargs['iou'](mgp_output, edge_GT)
            low_iou = kwargs['iou'](lowfreq, y)

            running_loss += total_loss.item()
            cnt += X.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {} | IoU low : {} | IoU High : {}".format(
                    epoch, batch_idx + 1, len(test_loader), np.round((batch_idx + 1) / len(test_loader) * 100.0, 2), avg_loss, low_iou, high_iou))

                pred = torch.sigmoid(mgp_output)
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                plot_test_results(X, resize(y), pred, epoch, batch_idx + 1)
                plot_mgp(X, spectrum, mask_list, k)

    return avg_loss

def fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, **kwargs):
    history = get_history()

    for epoch in tqdm(range(1, epochs + 1)) :
        start_time = time()

        print("TRAINING")
        train_loss = train_epoch(device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs)

        print("EVALUATE")
        test_loss = test_epoch(device, model, criterion, optimizer, test_loader, epoch, **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}| current_lr:{get_current_lr(optimizer)} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history