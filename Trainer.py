from time import time

from tqdm import tqdm

from utils import *
from utils.plot_functions import plot_test_results, plot_mgp

args = get_init()

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

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

        high_freq_output, low_freq_region, edge_GT = model(image, label, device)

        # LOSS
        high_edge_loss = criterion(high_freq_output.to(device), edge_GT.to(device))
        low_region_loss = criterion(low_freq_region.to(device), label)

        total_loss = kwargs['high_loss_weight'] * high_edge_loss + \
                     kwargs['low_loss_weight'] * low_region_loss

        running_loss += total_loss.item()
        cnt += image.size(0)

        avg_loss = running_loss / cnt

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}(HF loss : {} | LF loss : {})".format(
                epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2),
                running_loss / cnt, high_edge_loss, low_region_loss
            ))

    return avg_loss

def test_epoch(device, model, criterion, test_loader, epoch, **kwargs) :
    model.eval()
    running_loss, cnt = 0.0, 0
    avg_loss = 0.0
    for batch_idx, (image, label) in enumerate(test_loader):
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            high_freq_output, low_freq_region, edge_GT = model(image, label, device)

            # LOSS
            high_edge_loss = criterion(high_freq_output.to(device), edge_GT.to(device))
            low_region_loss = criterion(low_freq_region.to(device), label)

            total_loss = kwargs['high_loss_weight'] * high_edge_loss + \
                         kwargs['low_loss_weight'] * low_region_loss

            high_iou = iou_numpy(high_freq_output, edge_GT)
            low_iou = iou_numpy(low_freq_region, label)

            running_loss += total_loss.item()
            cnt += image.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {} | IoU low : {} | IoU High : {}".format(
                    epoch, batch_idx + 1, len(test_loader), np.round((batch_idx + 1) / len(test_loader) * 100.0, 2), avg_loss, low_iou, high_iou))

                pred = torch.sigmoid(low_freq_region)
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                plot_test_results(image, resize(label), pred, epoch, batch_idx + 1)
                # plot_mgp(image, spectrum, mask_list, k)

    return avg_loss

def fit(device, model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, **kwargs):
    history = get_history()

    for epoch in tqdm(range(1, epochs + 1)) :
        start_time = time()

        print("TRAINING")
        train_loss = train_epoch(device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs)

        print("EVALUATE")
        test_loss = test_epoch(device, model, criterion, test_loader, epoch, **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}| current_lr:{get_current_lr(optimizer)} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history