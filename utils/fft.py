import numpy as np
import json
from tqdm import tqdm

def FFT(coords):
    N = len(coords)
    phi = -np.pi / N
    A,F,P = [],[],[]  # Amplitude, Frequency, Phase
    temp = 0  # Calculationg percentage for pg window
    for i in tqdm(range(1 - N, N)):
        F = i
        ansX = 0
        ansY = 0
        for j in range(1 - N, N):
            ansX += coords[j][0] * np.cos(phi * i * j) - coords[j][1] * np.sin(phi * i * j)
            ansY += coords[j][1] * np.cos(phi * i * j) + coords[j][0] * np.sin(phi * i * j)

        ansX /= 2 * N
        ansY /= 2 * N
        A = np.sqrt(ansY ** 2 + ansX ** 2) # amplitude
        P = np.arctan2(ansY, ansX) + np.pi # phases
        A = np.append(A, A)
        P = np.append(P,P)
        # if temp != int((i + N) / (2 * N) * 100):
        #     temp = int((i + N) / (2 * N) * 100)
        #     screen.fill((0, 0, 0))
        #     medium_font = pg3.font.Font(font_loc, 20)
        #     loading_text = medium_font.render('Completed : ' + str(temp) + ' %', True, (0, 255, 255))
        #     screen.blit(loading_text, (0, 0))
        #     pg3.display.flip()
    return A, P

if __name__ == '__main__':
    import cv2
    import numpy as np
    img = cv2.imread('./img1.jpg',0)
    # A, P= FFT(img)

    # img_f = np.fft.fftshift(np.fft.fft2(img))
    # spectrum = np.abs(img_f)
    # phase = np.arctan2(img_f[, :], img_f[:, ])
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    phase_spectrum = np.angle(fshift)
    magnitude_spectrum = 20 * np.log(np.abs(fshift)+1)

    import matplotlib.pyplot as plt

    plt.imshow(phase_spectrum, cmap='gray')
    plt.show()