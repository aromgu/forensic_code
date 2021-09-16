import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic


n_seg = 300
img = cv2.imread('./2.jpg')
w,h,c = img.shape

segments = slic(img, start_label=1, n_segments = n_seg, sigma=0.2)
mask = np.zeros((w,h,1))
for i in range(5):
    random = np.random.randint(1,segments.max())
    zero_region = np.where(segments == random)
    mask[zero_region] = 1

plt.imshow(mask, cmap='gray')
plt.show()