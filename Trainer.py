from time import time

from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from utils import save_best_model
import configuration

resize = transforms.Compose(
    [transforms.Resize((256, 256))]
)

def train_epoch(device, model, criterion, optimizer, train_loader, epoch, **kwargs):
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
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        if kwargs['fad_option'] == 'y':
            kwargs['get_fad'].to('cuda')
            fad, lfs = kwargs['get_fad'](X)
            # fad = fad.resize(fad.size(0), fad.size(1), 256, 256)
            fad = resize(fad)

        optimizer.zero_grad()
        prediction = model(fad)
        pred = torch.sigmoid(prediction)

        if kwargs['patch_option'] == 'y':
            with torch.no_grad():
                gen_patch, proba = kwargs['patch_module'](X, y)
            kwargs['get_patchnet'].to('cuda')
            patch_pred = kwargs['get_patchnet'](gen_patch.float().to('cuda'))
            patch_loss = criterion(patch_pred, y)
        net_loss = criterion(pred, resize(y))

        # pixel_loss = criterion(proba.float(),classify_y.squeeze().float())
        total_loss = kwargs['net_loss_weight'] * net_loss + kwargs['patch_loss_weight'] * patch_loss  # + 0.2  * pixel_loss

        running_loss += total_loss.item()
        cnt += X.size(0)

        avg_loss = running_loss / cnt

        total_loss.backward()
        optimizer.step()

    return avg_loss


def test_epoch(device, model, criterion, optimizer, train_loader, epoch, **kwargs) :
    model.eval()
    running_loss, cnt = 0.0, 0
    avg_loss = 0.0
    patch_loss = 0.0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            if kwargs['fad_option'] == 'y':
                kwargs['get_fad'].to('cuda')
                fad, lfs = kwargs['get_fad'](X)
                fad = resize(fad)
            optimizer.zero_grad()
            prediction = model(fad)
            pred = torch.sigmoid(prediction)

            if kwargs['patch_option'] == 'y':
                gen_patch, proba = kwargs['patch_module'](X, y)
                kwargs['get_patchnet'].to('cuda')
                patch_pred = kwargs['get_patchnet'](torch.FloatTensor(gen_patch.float()).to('cuda'))
                patch_loss = criterion(patch_pred, y)

            net_loss = criterion(pred, resize(y))

            # pixel_loss = criterion(proba.float(),classify_y.squeeze().float())
            total_loss = kwargs['net_loss_weight'] * net_loss + kwargs['patch_loss_weight'] * patch_loss  # + 0.2  * pixel_loss

            running_loss += total_loss.item()
            cnt += X.size(0)

            avg_loss = running_loss / cnt  # 무야호 춧

    return avg_loss


def fit(scheduler, device, model, criterion, optimizer, train_loader, test_loader, epochs, **kwargs):
    history = dict()
    history['train_loss'] = list()
    history['test_loss'] = list()
    for epoch in tqdm(range(1, epochs + 1)) :
        start_time = time()

        train_loss = train_epoch(device, model, criterion, optimizer, train_loader, epoch, **kwargs)
        scheduler.step(train_loss)
        test_loss = test_epoch(device, model, criterion, optimizer, test_loader, epoch, **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        configuration.cur_cost = train_loss
        if configuration.pre_cost > configuration.cur_cost :
            print("pre cost", configuration.pre_cost)
            print("cur cost", configuration.cur_cost)
            # print(kwargs['parent_dir'] + f"{kwargs['model_name']}/{kwargs['patch_option']}")
            save_best_model(kwargs['parent_dir'], epoch, model, kwargs['model_name'], optimizer, train_loss, kwargs['fad_option'], kwargs['patch_option']) # best model을 저장하고
            configuration.pre_cost = configuration.cur_cost
        print(f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history