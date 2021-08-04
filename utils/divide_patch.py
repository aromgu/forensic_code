import os

import numpy
import torch
from utils.func import *
import torch.nn.functional as F
import math
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from utils.compute_gradients import normal_vec

def _extract_patches(X, patch_size, stride):
    patches = []
    h, w, c = X.shape
    X = np.transpose((X),(2,1,0))
    H = (h - patch_size) // stride
    W = (w - patch_size) // stride
    for i in range(0, H * stride, stride):
        for j in range(0, W * stride, stride):
            patch = X[i:i + patch_size, j:j + patch_size]
            patches.append(np.expand_dims(patch, axis=0))
    return patches, H, W, h, w

def divide_batch(X, y):
    b, c, h, w = X.shape
    batch_list = []
    y_list = [x for x in y]
    X = X.permute(0,3,2,1)
    for i in range(b):
        temp = X[i,:,:,:]
        batch_list.append(temp)

    tensor_list = []
    proba_list = []
    for i,j in zip(batch_list,y_list):
        i = i.cpu().detach().numpy()
        out, proba, y = localizing_window(i, j)
        # print('out', out.shape, 'pro', proba_list.shape, 'y', y.shape)
        # proba = pixel_probability(out)
        # tensor_list.append(out)
        out = np.expand_dims(out, axis=0)
        proba_list.append(proba)
        tensor_list.extend(out)
        # proba_list.append(proba)
    t_list = numpy.array(tensor_list)
    to_tensor = torch.from_numpy(t_list)
    proba_list = numpy.array(proba_list)
    proba_list = torch.from_numpy(proba_list)
    # print('to', to_tensor, 'proba', proba_list)
    return to_tensor, proba_list

def localizing_window(X, y):
    y = y
    # 320,384
    patch_size = 32
    stride = 32
    width = 256 #24
    height = 256 # 20
    # X = np.expand_dims(X.cpu().detach().numpy(), axis=3)
    patches, W, H, h, w = _extract_patches(X, patch_size=patch_size, stride=stride)
    bins = 10
    n_of_patches = H*W
    num_patches = math.ceil(len(patches) / n_of_patches) # batch_size

    for i in range(num_patches):
        if i == (num_patches - 1):  # last batch
            patch_x = patches[i * n_of_patches:]
        else:
            patch_x = patches[i * n_of_patches:(i + 1) * n_of_patches]

        # patch_x = np.array(patch_x)
        vec_list = np.array([])
        degree_list = []

        for i in range(len(patch_x)):
            vec2, degree = normal_vec(patch_x[i])
            vec_list = np.append(vec_list, np.array(vec2))
            degree_list.append(degree)

        # DEGREE HISTOGRAM ==
        # 히스토그램의 빈값과 bin_edged뽑기
        degree_list = np.where(np.isnan(degree_list), 0, degree_list)
        counts, bin_edges, patches = plt.hist(degree_list, bins=bins)
        # plt.title(f'bins: {bins}')
        # plt.show()
        sort_top_three = sorted(counts)
        # 뽑은 히스토그램중 TOP3 선별

        selected_three = sort_top_three[-3:]
        # 탑쓰리 값을 인덱스로 변환
        index = []
        for i in selected_three:
            if i in counts:
                index.extend(np.where(counts==i))
        index = np.unique(np.concatenate((index), axis=0))

        # 넘파이로 바꾸고 nan 값을 0 으로 대체
        degree_list = np.array(degree_list)
        degree_list=np.where(np.isnan(degree_list),0,degree_list)

        #  TOP3 히스토그램 구간 내 값들을 value 으로 반환
        def return_val(index):
            a = bin_edges[index][0]
            b = bin_edges[index+1][0]
            one = degree_list[(a <= degree_list) & (degree_list < b)]
            c = bin_edges[index][1]
            d = bin_edges[index+1][1]
            two = degree_list[(c <= degree_list) & (degree_list < d)]
            e = bin_edges[index][2]
            f = bin_edges[index+1][2]
            three = degree_list[(e <= degree_list) & (degree_list < f)]

            return one, two, three

        one, two, three = return_val(index)
    # import sklearn.metrics
    # import seaborn as sns
    # sns.set_theme(style="white")
    #
    # vec_list = np.reshape(vec_list,(-1,2))
    # cosine_similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(vec_list, Y=vec_list, dense_output=True)
    # f, ax = plt.subplots(figsize=(11, 9))
    #
    # sns.heatmap(1-cosine_similarity_matrix, cmap='YlGnBu', vmin=0, vmax=1, center=0,
    #             annot=False,
    #             square=True, linewidths=.5)# cbar_kws={"shrink": .5})
    # # plt.show()
    #
    # add_up = np.sum(cosine_similarity_matrix, axis=0)
    # get_vec = np.expand_dims(add_up, axis=1)
    #
    # get_vec = sum(list(map(list, get_vec)), [])
    # gt_map = np.ones((320, 384), dtype=np.uint8)
        gt_map = np.zeros((height, width), dtype=np.uint8)
        # height = 512, width = 1024

        # 탑3 값 합치기
        gt_list = np.concatenate((one,two,three), axis=0)
        # print('gt_list',gt_list)
        for k in range(len(degree_list)):
            # k = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ....., 31
            # 탑쓰리 리스트와 본 리스트 비교 후 마스크 생성
            for i in gt_list:
                if degree_list[k] == i:
                    i, j = divmod(k, W)
                    # i = 0, 1, 2, 3
                    # j = 0, 1, 2, 3, 4, 5, 6, 7
                    gt_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 1
        proba = pixel_probability(gt_map)

    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(gt_map, cmap='gray')
    # ax[0].set_title(f'pixel probability : {proba:.5f}')
    # # ax[1].imshow(cv2.cvtColor(X, cv2.COLOR_RGB2BGR))
    # ax[1].imshow(X, cmap='gray')
    # ax[2].imshow(y[0].transpose(1,2,0), cmap='gray')
    # plt.show()
        return np.expand_dims(gt_map, axis=0), proba, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target', type=str, default='img1.jpg')
    args = parser.parse_args()

    dir_name = './'
    file_name = args.target
    result_name = file_name.split('.')[0] + '_result.jpg'
    file_path = os.path.join(dir_name, file_name)
    result_path = os.path.join(dir_name, result_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    im = Image.open(file_path)
    im = im.convert('L')
    x = np.array(im)#[:, :, 0]

    # result, y = localizing_window(x, y)
