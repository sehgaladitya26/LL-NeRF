from dual_channel import *
from basic_prep import *
import os 
import cv2 as cv
import matplotlib.pyplot as plt

path = './data/'
dest = './results/GamCor/'
files = os.listdir(path)
for file in files:
    if file.endswith('.jpg'):
        I = cv.imread(path+file)
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
        #I = cv.resize(I, (256, 256))
        J = Gamma_Correction(I)
        #J = HistEqualizer_HSV(I)
        #J = Contrast_Enhancer(I)
        #J = Brightness_Enhancer(I)
        #J = dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False)
        J = denoiser(J)
        plt.imsave(dest+file, J)
    elif file.endswith('.png'):
        I = cv.imread(path+file)
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
        I = cv.resize(I, (256, 256))
        J = dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False)
        J = denoiser(J)
        plt.imsave(dest+file, J)
    elif file.endswith('.jpeg'):
        I = cv.imread(path+file)
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
        I = cv.resize(I, (256, 256))
        J = dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False)
        J = denoiser(J)
        plt.imsave(dest+file, J)