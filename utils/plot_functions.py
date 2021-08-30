import os

import numpy as np
import matplotlib.pyplot as plt

def plot_test_results(image, ground_truth, prediction, epoch, batch_idx, save_root_path='example') :
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.transpose(image[0].squeeze().cpu().detach().numpy(), (1, 2, 0)))
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[0].imshow(ground_truth[0].squeeze().cpu().detach().numpy(), cmap='inferno', interpolation='none', alpha=0.5)
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[1].imshow(prediction[0].squeeze().cpu().detach().numpy(), cmap='gray')
    ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    if not os.path.exists(os.path.join(save_root_path, "epoch{}".format(str(epoch)))) :
        os.makedirs(os.path.join(save_root_path, "epoch{}".format(str(epoch))))

    plt.savefig(os.path.join(save_root_path, "epoch{}/example_{}.png".format(str(epoch), str(batch_idx))),
                bbox_inches='tight', pad_inches=0)

    plt.close()

def plot_loss(history) :
    train_loss = history['train_loss']
    test_loss = history['test_loss']

    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss', color='r')
    plt.plot(np.arange(len(test_loss)), test_loss, label='test loss', color='skyblue')
    plt.legend()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight', pad_inches=0)
# import torch
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from models import *
# from utils import *
# from utils.data_loader import load_path
#
# args = get_init()
# device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
# PATH = '../saved_models/' + f'{args.model_name}/{args.patch_option}/best2.pth'
# # model = torch.load(PATH)
# model = nn.DataParallel(get_model(args.model_name))
# model.load_state_dict(torch.load(PATH)['model_state_dict'])
# train_loader, test_loader, infer_loader = load_path(Au_image_path='../'+args.Au_image_path,
#                                                     Tp_image_path='../'+args.Tp_image_path,
#                                                     Tp_label_path='../'+args.Tp_label_path, split_ratio=args.split_ratio,
#                                                     batch_size=args.batch_size)
#
# # def inference(model, infer_loader):
# model.eval()
# for X, y in infer_loader:
#     X = X.to(device)
#     y = y.to(device)
#     with torch.no_grad():
#         prediction = model(X)
#         pred = torch.sigmoid(prediction)
#     # prediction = [B, output_dim, H, W]
#     # print(torch.max(pred))
#     predict = torch.argmax(pred, dim=1)
#
#     raw = X[0].permute(1,2,0).cpu().detach().numpy()#.astype(np.uint8)
#     # raw = cv2.cvtColor(raw.cpu().detach().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
#
#     predict = cv2.cvtColor(predict.permute(1,2,0).cpu().detach().numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
#     y = cv2.cvtColor(y[0].permute(1,2,0).cpu().detach().numpy().astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)
#     # summation = cv2.addWeighted(raw, 1, y, 0.4, 1)
#     fig, ax = plt.subplots(1,3)
#     ax[0].imshow(raw)
#     ax[1].imshow(y, cmap='gray')
#     ax[2].imshow(predict*255)
#     plt.show()
    # final = np.concatenate([raw, y, predict*255], axis=1)
    # cv2.imshow('final', final)
    # cv2.waitKey(150)

# focal loss
# if __name__ == '___main__':


# inference(PATH, model, args.model_name, args.epoch, infer_loader)

