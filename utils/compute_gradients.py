from __future__ import division
import cv2
import matplotlib.pyplot as plt
from utils.func import *
# from mask_select import *
import argparse
import numpy as np
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
# from dataloader import CreateDataset, split
import torch

ROOT = './dataset'
def normal_vec(img):
    # img = img.squeeze().permute(2,1,0)
    # img = img.numpy().cpu()
    # h, w,_ = img.shape
    h, w, c = img[0].shape
    # img = img[0]
    if h == w:
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.expand_dims(img, axis=2)
        img = np.float32(img[0])

        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        med = cv2.medianBlur(grayscale, 5)

        bilatfilter = cv2.bilateralFilter(med, 9, 75, 75)
        x_sobel = cv2.Sobel(bilatfilter, cv2.CV_32F, 1, 0, 3)
        y_sobel = cv2.Sobel(bilatfilter, cv2.CV_32F, 0, 1, 3)
        sobel = cv2.Sobel(bilatfilter, cv2.CV_32F, 1, 1, 3)

        norm1 = norm(sobel)

        # Threshold =============================
        nonzero = np.count_nonzero(norm1)

        mean = np.sqrt(np.sum(sobel**2)) / nonzero
        sigma = np.sqrt((np.sqrt(np.sum(sobel**2)) - mean)**2 / nonzero)
        t = mean+sigma

        # _, threshed = cv2.threshold(sobel, t, 0, cv2.THRESH_TOZERO_INV)
        #
        # # Threshold =============================
        # _, x_threshed = cv2.threshold(x_sobel, t, 0, cv2.THRESH_TOZERO_INV)
        # _, y_threshed = cv2.threshold(y_sobel, t, 0, cv2.THRESH_TOZERO_INV)

        # B. Lighting Representation and Dissimilarity Features =============================
        # magnitude-normalized gradient vectors
        norm2 = norm(sobel)

        # dI^ hat eq.(15)
        di_x = np.transpose(x_sobel / norm1)
        di_y = np.transpose(y_sobel / norm1)
        # di_hat = np.transpose(sobel / norm1)
        # di_hat = np.transpose(np.dot(di_x,di_y))
        # dI- eq.14
        sum_x0 = np.transpose(np.sum(di_x) / (h * w))
        sum_y0 = np.transpose(np.sum(di_y) / (h * w))
        # for cos_dis.14
        # di_bar = np.transpose(np.sum(di_hat) / (h * w))

        #### Quiver
        # plt.figure(figsize=(3,3))
        # plt.tight_layout()
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        # # h,w=masked_r.shape
        # # linx, liny = np.meshgrid(np.linspace(-3, 3, 1), np.linspace(-3, 3, 1))
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # # plt.set_title('Authentic image')
        # # print(sum_x0, sum_y0)
        # plt.quiver(w/2, h/2, sum_y0*1e5, sum_x0*1e5, width=0.01, scale=500, color='red', label='Lightning vector')
        # plt.legend(loc='lower left')
        # plt.show()
        # print(sum_x0, sum_y0)

    # Characterize gradient vector field
    #     sobel_R = sobel_r[sobel_r>0]

        # scaler = MinMaxScaler()

        hist, bins = np.histogram(sobel.ravel(), 72)
        his_mean = hist/72

        # Compute degree
        a = [1,0]
        sim = spatial.distance.cosine(a,[sum_x0,sum_y0])
        # sim = cos_sim(a, (sum_x0,sum_y0))
        arccos = np.arccos(sim)
        degree = np.degrees(arccos)
        # print('de',degree)
        return (sum_x0, sum_y0), degree
    else: return (np.nan,np.nan), np.nan



# if __name__ == "__main__":
#     img_path = './datasets/casia2groundtruth/CASIA2.0_revised/Au/Au_ani_00001.jpg'
#     img = cv2.imread(img_path, 0)
#     normal_vec(img)
