from time import time

import torch
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm

import configuration
from utils.save_functions import save_best_model
from utils.get_functions import get_current_lr
from utils.plot_functions import plot_test_results

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
    patch_loss = 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        if kwargs['fad_option'] == 'y':
            kwargs['get_fad'].to('cuda')
            fad, lfs = kwargs['get_fad'](X)
            # fad = fad.resize(fad.size(0), fad.size(1), 256, 256)
            fad = resize(fad)

        optimizer.zero_grad()
        prediction = model(X)

        if kwargs['patch_option'] == 'y':
            with torch.no_grad():
                gen_patch, proba = kwargs['patch_module'](X, y)
            kwargs['get_patchnet'].to('cuda')
            patch_pred = kwargs['get_patchnet'](gen_patch.float().to('cuda'))
            patch_loss = criterion(patch_pred, y)
        net_loss = criterion(prediction, resize(y)) # # BCEWithLogitsLoss는 sigmoid가 추가된 손실함수이기 때문에 따로 sigmoid를 추가적으로 해줄 필요 없음

        # pixel_loss = criterion(proba.float(),classify_y.squeeze().float())
        # total_loss = kwargs['net_loss_weight'] * net_loss + kwargs['patch_loss_weight'] * patch_loss  # + 0.2  * pixel_loss

        running_loss += net_loss.item()
        cnt += X.size(0)

        avg_loss = running_loss / cnt

        net_loss.backward()
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
    patch_loss = 0.0
    for batch_idx, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            if kwargs['fad_option'] == 'y':
                kwargs['get_fad'].to('cuda')
                fad, lfs = kwargs['get_fad'](X)
                fad = resize(fad)

            optimizer.zero_grad()
            prediction = model(X)
            # pred = torch.sigmoid(prediction)

            if kwargs['patch_option'] == 'y':
                gen_patch, proba = kwargs['patch_module'](X, y)
                kwargs['get_patchnet'].to('cuda')
                patch_pred = kwargs['get_patchnet'](torch.FloatTensor(gen_patch.float()).to('cuda'))
                patch_loss = criterion(patch_pred, y)

            net_loss = criterion(prediction, resize(y))

            # pixel_loss = criterion(proba.float(),classify_y.squeeze().float())
            total_loss = kwargs['net_loss_weight'] * net_loss + kwargs['patch_loss_weight'] * patch_loss  # + 0.2  * pixel_loss

            running_loss += total_loss.item()
            cnt += X.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print("Epoch {}  | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(test_loader), np.round((batch_idx + 1) / len(test_loader) * 100.0, 2), avg_loss))

                pred = torch.sigmoid(prediction)
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                plot_test_results(X, resize(y), pred, epoch, batch_idx + 1)

    return avg_loss

def fit(scheduler, device, model, criterion, optimizer, train_loader, test_loader, epochs, **kwargs):
    history = dict()
    history['train_loss'] = list()
    history['test_loss'] = list()
    for epoch in tqdm(range(1, epochs + 1)) :
        start_time = time()

        print("TRAINING")
        train_loss = train_epoch(device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs)
        # scheduler.step(train_loss) # StepLR 스케쥴러는 여기있는게 맞지만 Cosine annealing scheduler는 매 iter마다 step을 수행

        print("EVALUATE")
        test_loss = test_epoch(device, model, criterion, optimizer, test_loader, epoch, **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        configuration.cur_cost = test_loss
        if configuration.pre_cost > configuration.cur_cost :
            print("pre cost", configuration.pre_cost)
            print("cur cost", configuration.cur_cost)
            # print(kwargs['parent_dir'] + f"{kwargs['model_name']}/{kwargs['patch_option']}")
            save_best_model(kwargs['parent_dir'], epoch, model, kwargs['model_name'], optimizer, train_loss, kwargs['fad_option'], kwargs['patch_option']) # best model을 저장하고
            configuration.pre_cost = configuration.cur_cost
        print(f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}| current_lr:{get_current_lr(optimizer)} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history