import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def Contrast_Enhancer(input, alpha=2, beta=10):
    img = input.copy()
    # alpha  = Contrast control (1.0-3.0)
    # beta = Brightness control (0-100)
    adjusted = cv.convertScaleAbs(img, alpha, beta)
    return adjusted.astype(np.uint8)


def Brightness_Enhancer(input, cl=2.0, gs=8):
    img = input.copy()
    # converting to LAB color space
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=cl, tileGridSize=(gs,gs))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return enhanced_img.astype(np.uint8)


def HistEqualizer_HSV(input):
    img = input.copy()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = cv.equalizeHist(v)
    hsv = cv.merge((h, s, v))
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR).astype(np.uint8)

def Gamma_Correction(input, gamma=2):
    img = input.copy()
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table).astype(np.uint8)


def denoiser(input, sigma=0.5):
    img = input.copy()
    return cv.fastNlMeansDenoisingColored(img, None, sigma, 10, 7, 21).astype(np.uint8)

# input = cv.imread('./data/ezgif-frame-001.jpg')
# input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
# o1 = Contrast_Enhancer(input)
# o1 = denoiser(o1)
# o2 = Brightness_Enhancer(input)
# o2 = denoiser(o2)
# o3 = HistEqualizer_HSV(input)
# o3 = denoiser(o3)
# o4 = Gamma_Correction(input)
# o4 = denoiser(o4)
# plt.imshow(o4)
# plt.show()

