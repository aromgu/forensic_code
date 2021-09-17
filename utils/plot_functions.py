import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from utils.get_functions import get_init

args = get_init()

def plot_test_results(image, ground_truth, pred,  epoch, batch_idx, save_root_path=args.save_root) :
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(np.transpose(image[0].squeeze().cpu().detach().numpy(), (1, 2, 0)))
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_title('Input image')

    ax[1].imshow(ground_truth[0].squeeze().cpu().detach().numpy(), cmap='gray')
    ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_title('Ground Truth')

    ax[2].imshow(pred[0].squeeze().cpu().detach().numpy(), cmap='gray')
    ax[2].axis('off'); ax[2].set_xticks([]); ax[2].set_yticks([]) #ax[2].set_title('HighPred')

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    ax[3].imshow(pred[0].squeeze().cpu().detach().numpy(), cmap='gray')
    ax[3].axis('off'); ax[3].set_xticks([]); ax[3].set_yticks([]) #ax[2].set_title('HighPred')
    # ax[3].imshow(low_pred[0].squeeze().cpu().detach().numpy(), cmap='gray')
    # ax[3].axis('off'); ax[3].set_xticks([]); ax[3].set_yticks([]);ax[3].set_title('LowPred')

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    if not os.path.exists(os.path.join('plot',save_root_path, "epoch{}".format(str(epoch)))) :
        os.makedirs(os.path.join('plot',save_root_path, "epoch{}".format(str(epoch))))

    plt.savefig(os.path.join('plot', save_root_path, "epoch{}/example_{}.png".format(str(epoch), str(batch_idx))),
                bbox_inches='tight', pad_inches=0)

    plt.close()

def plot_inputs(image, ground_truth, pred, epoch, batch_idx, save_root_path=args.save_root) :
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(np.transpose(image[0].squeeze().cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
    ax[0].imshow(ground_truth[0].squeeze().cpu().detach().numpy(), cmap='inferno', interpolation='none', alpha=0.6)
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_title('Input')

    ax[1].imshow(pred[0].squeeze().cpu().detach().numpy(), cmap='gray')
    ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_title('pred')


    # ax[2].imshow(low_freq_output[0].squeeze().cpu().detach().numpy(), cmap='gray')
    # ax[2].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[2].set_title('LowInput')


    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    if not os.path.exists(os.path.join(save_root_path, "epoch{}".format(str(epoch)))) :
        os.makedirs(os.path.join(save_root_path, "epoch{}".format(str(epoch))))

    plt.savefig(os.path.join(save_root_path, "epoch{}/example_{}.png".format(str(epoch), str(batch_idx))),
                bbox_inches='tight', pad_inches=0)

    plt.close()

def plot_mgp(X, spectrum, mask_list, k):
    fig, ax = plt.subplots(2, k)
    ax[0, 0].imshow(X[0].squeeze().cpu().detach().numpy(), cmap='gray')
    ax[0, 0].set_title('Input');ax[0, 0].set_xticks([]);ax[0, 0].set_yticks([])
    ax[0, 1].imshow(spectrum[0].squeeze().cpu().detach().numpy(), cmap='gray')
    ax[0, 1].set_title('Spectrum');ax[0, 1].set_xticks([]);ax[0, 1].set_yticks([])
    ax[0, 2].axis('off')

    for i in range(k):
        ax[1, i].imshow(mask_list[i].squeeze().cpu().detach().numpy(), cmap='gray')
        ax[1, i].set_xticks([]);ax[1, i].set_yticks([])
        ax[0, i].axis('off')
    ax[1, 0].set_title('MGP masks');ax[1, 0].set_xticks([]);ax[1, 0].set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('./MGP_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_loss(parent_dir, history, save_root) :
    train_loss = history['train_loss']
    test_loss = history['test_loss']

    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss', color='r')
    plt.plot(np.arange(len(test_loss)), test_loss, label='test loss', color='skyblue')
    plt.title('Loss Tendency')
    plt.legend()
    plt.ylim(0,3)
    # plt.show()

    plt.savefig(os.path.join(parent_dir, save_root,'loss.png'), dpi=300, bbox_inches='tight', pad_inches=0)


def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

if __name__ == '__main__':
    f = open('./history.txt', 'r').readlines()
    plot_loss('./loss.png', f, 'lol')