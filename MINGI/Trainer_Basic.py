from time import time
from tqdm import tqdm
from utils import *
from utils.plot_functions import plot_test_results

def train_epoch(args, device, model, criterion, optimizer, scheduler, train_loader, epoch, max_norm=1.0, **kwargs):
    model.train()
    running_loss, cnt = 0.0, 0

    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        output = model(image)

        loss = criterion(output, label)

        running_loss += loss.item()
        cnt += image.size(0)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader) :
            print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | Running Loss : {}".format(
                args.model_name, epoch, batch_idx + 1, len(train_loader), np.round((batch_idx + 1) / len(train_loader) * 100.0, 2), round(loss.item(), 4)
            ))

    avg_loss = running_loss / cnt

    return avg_loss

def test_epoch(args, device, model, criterion, test_loader, epoch, **kwargs) :
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

            output = model(image)
            loss = criterion(output, label)

            running_loss += loss.item()
            cnt += image.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print("{} Epoch {} | batch_idx : {}/{}({}%) COMPLETE | Running Loss : {}".format(
                    args.model_name, epoch, batch_idx + 1, len(test_loader),
                    np.round((batch_idx + 1) / len(test_loader) * 100.0, 2), round(loss.item(), 4)
                ))

                plot_test_results(image, label, output, epoch, batch_idx + 1, [0.0, 0.0, 0.0, 0.0], save_root_path=args.save_root)

            if epoch == args.epochs:
                pred = output
                auc, precision, recall, f1, iou = get_metrices(label, pred)
                auc_list.append(auc)
                f1_list.append(f1)
                iou_list.append(iou)
                precision_list.append(precision)
                recall_list.append(recall)

                plot_test_results(image, label, pred, epoch, batch_idx + 1, [f1, iou, precision, recall], save_root_path=args.save_root)

    avg_loss = running_loss / cnt

    return round(avg_loss,4), round(np.mean(auc_list),4), round(np.mean(iou_list),4), \
           round(np.mean(f1_list),4), round(np.mean(precision_list),4), round(np.mean(recall_list),4)

def fit(args, device, model, criterion, optimizer, scheduler, train_loader, test_loader, **kwargs):
    history = get_history()
    print_configurations(args)
    auc, iou, f1, precision, recall = 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch in tqdm(range(1, args.epochs + 1)):
        start_time = time()

        print("TRAINING")
        train_loss = train_epoch(args, device, model, criterion, optimizer, scheduler, train_loader, epoch, **kwargs)

        print("EVALUATE")
        test_loss, auc, iou, f1, precision, recall = test_epoch(args, device, model, criterion, test_loader, epoch,
                                                                **kwargs)

        end_time = time() - start_time
        m, s = divmod(end_time, 60)
        h, m = divmod(m, 60)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        print(
            f'epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}| current_lr:{get_current_lr(optimizer)} | took : {int(h)}h {int(m)}m {int(s)}s')

    return model, history, auc, iou, f1, precision, recall