from preprocessing import dehaze
import os 
import cv2 as cv
import matplotlib.pyplot as plt

path = './data/'
dest = './results/'
files = os.listdir(path)
for file in files:
    if file.endswith('.jpg'):
        I = cv.imread(path+file)
        J = dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False)
        plt.imsave(dest+file, J)
    elif file.endswith('.png'):
        I = cv.imread(path+file)
        J = dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False)
        plt.imsave(dest+file, J)
    elif file.endswith('.jpeg'):
        I = cv.imread(path+file)
        J = dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False)
        plt.imsave(dest+file, J)