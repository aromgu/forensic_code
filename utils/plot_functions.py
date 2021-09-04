import os

import numpy as np
import matplotlib.pyplot as plt

def plot_test_results(image, ground_truth, prediction, epoch, batch_idx, save_root_path='example') :
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.transpose(image[0].squeeze().cpu().detach().numpy(), (1, 2, 0)))
    ax[0].imshow(ground_truth[0].squeeze().cpu().detach().numpy(), cmap='inferno', interpolation='none', alpha=0.6)
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[1].imshow(np.transpose(image[0].squeeze().cpu().detach().numpy(), (1, 2, 0)))
    ax[1].imshow(prediction[0].squeeze().cpu().detach().numpy(), cmap='inferno', interpolation='none', alpha=0.6)
    ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([])

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

def plot_loss(history, model_name, fad_option) :
    train_loss = history['train_loss']
    test_loss = history['test_loss']

    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss', color='r')
    plt.plot(np.arange(len(test_loss)), test_loss, label='test loss', color='skyblue')
    plt.legend()
    plt.ylim(0,2)

    plt.savefig(f'{model_name}''_fad_'f'{fad_option}''.png', dpi=300, bbox_inches='tight', pad_inches=0)