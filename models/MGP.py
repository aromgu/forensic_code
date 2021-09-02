import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.fft as fft
import cv2

def fourier_intensity_extraction(x_spectrum, idxx, idxy, num_clusters, image_size) :
    x_spectrum = x_spectrum.cpu().detach().numpy()
    data = []
    for idxx_, idxy_ in zip(idxx, idxy):
        data.append([idxx_, idxy_, np.sum(x_spectrum[..., idxx_, idxy_])])

    num_pix = []
    sum_int = []

    for i in range(len(data)):
        if len(data[i][0]) < 1 : continue
        num_pix.append(len(data[i][0]))
        sum_int.append(data[i][2])

    num_pix = np.array(num_pix).reshape(-1, 1)
    sum_int = np.array(sum_int).reshape(-1, 1)

    X = sum_int / num_pix
    km = KMeans(n_clusters=num_clusters, random_state=4321)
    km.fit(X)
    labels = km.predict(X)

    data_np = np.array(data)
    clustering_data = []

    for label in np.unique(labels):
        clustering_data.append(data_np[np.where(labels == label)[0].tolist()])

    clustered_idx = []
    for label in np.unique(labels):
        mask = np.zeros((image_size, image_size))
        for label_ in clustering_data[label] :
            idxx, idxy = label_[0], label_[1]
            mask[idxx, idxy] = 1
        clustered_idx.append(np.where(mask == 1))

    return clustered_idx, X, labels


def get_small_region(image_size, angle, length, preserve_range):
    idxx, idxy = [], []
    start = preserve_range
    x_range = np.arange(0, image_size) - int(image_size / 2)
    y_range = np.arange(0, image_size) - int(image_size / 2)

    x_ms, y_ms = np.meshgrid(x_range, y_range)

    R = np.sqrt(x_ms ** 2 + y_ms ** 2)
    T = np.degrees(np.arctan2(y_ms, x_ms))

    T[T < 0] += 360

    for l in range(start, image_size // 2, length):
        # print(l)
        for d in range(0, 360, angle):
            idxx_, idxy_ = get_small_region_idx(image_size, angle, length, R, T, l, d)
            idxx.append(idxx_); idxy.append(idxy_)

    return idxx, idxy

def get_small_region_idx(image_size, angle, length, R, T, l, d):
    if l + length <= image_size // 2 : idxx, idxy = np.where((R > l) & (R <= l + length) & (T >= d) & (T < d + angle))
    else : idxx, idxy = np.where((R > l) & (R <= image_size//2) & (T >= d) & (T < d + angle))

    return idxx, idxy

def extract(x, device, image_size, angle, length, preserve_range, num_enc):
    idxx, idxy = get_small_region(image_size, angle, length, preserve_range)

    x_fft = fft.fftshift(fft.fft2(x))
    x_spectrum = torch.log(1 + torch.abs(x_fft))
    clustered_idx, X, labels = fourier_intensity_extraction(x_spectrum, idxx, idxy, num_enc, image_size)
    patterns = torch.empty((num_enc, x_fft.size(0), x_fft.size(2), x_fft.size(3))).to(device, dtype=torch.float32)

    mask_list = []
    for i, (idxx, idxy) in enumerate(clustered_idx):
        mask = torch.zeros(x_fft.size(2), x_fft.size(3)).to(device)
        mask[idxx, idxy] = 1
        temp = torch.empty_like(x_fft)
        mask_list.append(mask)
        for j in range(len(x_fft)):
            temp[j] = x_fft[j] * mask
            temp[j] = torch.abs(fft.ifft2(fft.ifftshift(temp[j])))

        patterns[i] = temp.squeeze()

    return patterns, x_spectrum, mask_list

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


def fft_(input):
    f = np.fft.fft2(input)
    shift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(shift))
    return shift, magnitude_spectrum

def idft(input):
    ishift = np.fft.ifftshift(input)
    out = np.fft.ifft2(ishift)
    out = np.abs(out)
    return out

def lowfreq_mask(radius, src):
    b,c,w,h = src.shape
    src = src.cpu().detach().numpy()
    circle_mask = circular(h, w, radius)

    zeros = torch.zeros(b,c,w,h)
    for i in range(b):
        one_img = src[i,:,:,:]
        one_img, _ = fft_(one_img)
        dst = one_img*(1-circle_mask)
        dst = idft(dst)
        zeros[i] = torch.from_numpy(dst)
    return zeros


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

    src_hi = cv2.imread('../utils/img1.jpg')
    ycbcr = cv2.cvtColor(src_hi, cv2.COLOR_RGB2YUV)
    resize = cv2.resize(ycbcr, (256,256))
    src = torch.unsqueeze(torch.from_numpy(resize[:,:,0]), dim=0).to(device)
    cir = circular(256,256, 15)
    plt.imshow(cir, cmap='gray')
    plt.show()
    # patterns = extract(src, device, 256, 20, 20, 0, 5)

    # merge = np.zeros((256,256,3))
    # merge[:,:,0] = patterns[4].squeeze().cpu().detach().numpy()
    # merge[:,:,1] = resize[:,:,1]
    # merge[:,:,2] = resize[:,:,2]

    # rgb = cv2.cvtColor(np.uint8(merge), cv2.COLOR_YUV2BGR)

    # ax[0, 0].imshow(rgb.astype(np.uint8))
    # ax[0, 1].imshow(patterns[0].squeeze().cpu().detach().numpy(), cmap='gray')
    # ax[0, 2].imshow(patterns[1].squeeze().cpu().detach().numpy(), cmap='gray')
    # ax[1, 0].imshow(patterns[2].squeeze().cpu().detach().numpy(), cmap='gray')
    # ax[1, 1].imshow(patterns[3].squeeze().cpu().detach().numpy(), cmap='gray')
    # ax[1, 2].imshow(patterns[4].squeeze().cpu().detach().numpy(), cmap='gray')
    # plt.show()