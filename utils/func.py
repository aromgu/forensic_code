import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import gridspec
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

import numpy as np
import json
from tqdm import tqdm

def pixel_probability(X):
    h, w = X.shape
    proba = np.count_nonzero(X) / (h*w)
    return proba

def super_pixel(image, sum_seg):
    sum_seg = 21
    segments = slic(image, n_segments = sum_seg, sigma = 10)
    seg_list = []
    for i in range(len(segments)):
        zero_region = np.where(segments == i)
        mask = np.zeros_like(image)
        mask[zero_region] = 1
        seg_list.append(mask*image)
    return seg_list

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def plot_shit(list):
    fig, ax = plt.subplots(4, 8, figsize=(15, 6))
    for i in range(len(list)):
        row, col = divmod(i, 8)
        # i = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 .... 31
        # row = 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1
        # col = 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
        fig.tight_layout()
        ax[row][col].imshow(list[i], cmap='gray')
        ax[row][col].get_xaxis().set_visible(False)
        ax[row][col].get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.show()

def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 15
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    res = []
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        res.append(fimg)
        # np.maximum(accum, fimg, accum)
    return res

def cos_dis(a,b):
    c = np.dot(np.transpose(a),b) / np.dot(np.sqrt(np.sum(a**2)),np.sqrt(np.sum(b**2)))
    if np.isnan(c) == True:
        c = 1.0
    ld = 1 - ((c + 1.0) / 2)
    return ld

def cos_sim(a,b):
    res = np.dot(a,b) / (norm(a)*norm(b))
    return res

def norm(img):
    return np.sqrt(np.sum(img**2))

def data_split(split, ratio=0.3):
    idx = np.arange(0, len(split))
    np.random.shuffle(idx)
    length = int(len(split) * ratio)
    train_data = split[length:]
    test_data = split[:length]
    return train_data, test_data

def z_score_norm(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized


def ycbcr(rgb_img):
    rgb_img = np.array(rgb_img)
    rgb_img = rgb_img.astype(np.uint8)
    im_ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCR_CB)
    y, cr, cb = cv2.split(im_ycrcb)
    return y, cb, cr

def fft(input):
    f = np.fft.fft2(input)
    shift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(shift))
    return shift, magnitude_spectrum

def delog(input):
    delog = np.exp(input / 20)
    return delog

def idft(input):
    ishift = np.fft.ifftshift(input)
    out = np.fft.ifft2(ishift)
    out = np.abs(out)
    return out

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def circular (h, w, num):
    c_mask = create_circular_mask(h, w)
    # masked_img = image.copy()
    # masked_img[~c_mask] = 0
    center = [int(w / 4), int(h / 4)]
    c_mask.reshape(h, w)
    c_mask = create_circular_mask(h, w, center=center)
    radius = h / num
    c_mask = create_circular_mask(h, w, radius=radius)
    a = np.zeros((h,w))
    c_mask = a-c_mask
    return c_mask

def in_circular (h, w, num, image):
    c_mask = create_circular_mask(h, w)
    # masked_img = image.copy()
    # masked_img[~c_mask] = 0
    center = [int(w / 4), int(h / 4)]
    c_mask.reshape(h, w)
    c_mask = create_circular_mask(h, w, center=center)
    radius = h / num
    c_mask = create_circular_mask(h, w, radius=radius)
    a = np.zeros((h,w))
    c_mask = a-c_mask
    return 1-c_mask

def gaussian_filter(w, h, sx, sy):
    sigmax, sigmay = sx, sy
    cx, cy = w / 2, h / 2
    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X - cx) / sigmax) ** 2 + ((Y - cy) / sigmay) ** 2))
    return gmask

def horizon (rows, cols, thick):
    plane1 = np.zeros((rows, cols))
    plane2 = np.zeros((rows, cols))
    crow, ccol = (int)(rows / 2), (int)(cols / 2)
    plane1[crow-1:crow+1, 24:ccol -1] = 1
    plane2[crow-1:crow+1, ccol+1:cols-24] = 1

    A = cv2.getRotationMatrix2D((ccol, crow),45, 4.0)
    B = cv2.getRotationMatrix2D((ccol, crow), 225, 4.0)
    C = cv2.getRotationMatrix2D((ccol, crow),0, 4.0)
    D = cv2.getRotationMatrix2D((ccol, crow), 180, 4.0)
    # C = cv2.getRotationMatrix2D((ccol, crow), 215, 1.0)
    rotated1 = cv2.warpAffine(plane1, A, (cols, rows))
    rotated2 = cv2.warpAffine(plane1, B, (cols, rows))
    rotated3 = cv2.warpAffine(plane1, C, (cols, rows))
    rotated4 = cv2.warpAffine(plane1, D, (cols, rows))
    # rotated3 = cv2.warpAffine(plane1, C, (cols, rows))

    mask = rotated1+rotated2+rotated3+rotated4
    omask = 1-mask
    return mask

# def horizon (rows, cols, thick):
#     plane1 = np.ones((rows, cols))
#     plane2 = np.ones((rows, cols))
#     crow, ccol = (int)(rows / 2), (int)(cols / 2)
#     plane1[crow-3:crow+3, 150:ccol -5] = 0
#     plane2[crow-3:crow+3, ccol+5:cols-150] = 0
#     mask = plane1+plane2
#     return mask

def vertical(h, w, thick):
    plane1 = np.ones((h, w))
    plane2 = np.ones((h, w))
    ch, cw = (int)(h / 2), (int)(w / 2)
    plane1[0:ch, cw-thick:cw+thick] = 0
    plane2[ch:h, cw-thick:cw+thick] = 0
    mask = 1-plane2*plane1
    return mask

# def try1(w, h):
#     plane2 = np.ones((w, h))
#     plane3 = np.ones((w, h))
#     plane2[0:145, 100:120] = 0
#     plane3[160:w, 150:170] = 0
#     # plane2 = 1-plane2
#     mask = plane2+plane3
#     return mask

def try1(h, w):
    white_color=(255,255,255)
    plane = np.zeros((h, w))
    ch, cw = (int)(h / 2), (int)(w / 2)

    points1 = np.array([[0, ch-10], [cw+6, ch], [0, ch+10]])
    points2 = np.array([[w, ch-10], [cw-6, ch], [w, ch+10]])

    img1 = cv2.fillPoly(plane,[points1],white_color)
    img2 = cv2.fillPoly(plane, [points2], white_color)

    mask = points1

    return plane

def rgb(y, cr, cb):

    temp = np.zeros((y.shape[0], y.shape[1], 3), dtype=np.uint8)
    temp[:,:,0] = y
    temp[:,:,1] = cr
    temp[:,:,2] = cb
    im_ycbcr = temp.astype(np.uint8)
    # im_ycbcr = cv2.merge([y, cr, cb])
    # im_ycbcr = im_ycbcr.astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycbcr, cv2.COLOR_YCR_CB2RGB)
    r, g, b = cv2.split(im_rgb)
    return b, g, r

import math

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def pixel(mask):
    a = np.count_nonzero(mask)
    return a