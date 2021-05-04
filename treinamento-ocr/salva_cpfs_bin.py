from PIL import ImageFont, ImageDraw, Image
import cv2
import time
import numpy as np
import skimage.measure
import os
import glob

#torna imagens binarias
def settozero(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO)
    return thresh

def sharpenFunction(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #matriz de convolução que aplica nitidez
    return cv2.filter2D(image, -1, kernel)

def filterFunction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T,Thresh1) = cv2.threshold(gray, 187, 255, cv2.THRESH_TRUNC)
    Thresh2 = cv2.adaptiveThreshold(Thresh1, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 231,50)
    return Thresh2

im_folder = os.listdir("ocr-treino/num_cortes/")
cont = 0

for p in im_folder:
    i = 0
    for c in glob.glob(f"ocr-treino/num_cortes/{cont}/*.jpg"):
        img = cv2.imread(c)
        nitida = sharpenFunction(img)
        filt = filterFunction(nitida)
        blur = cv2.medianBlur(filt, 3)
        cv2.imshow('bin'+ str(cont), blur)
        k = cv2.waitKey(60)
        if k == ord('q'):
            exit()
        blur = cv2.resize(blur, (30,50))
        #print(blur.shape)
        cv2.imwrite(f"ocr-treino/num_cortes_bin/{cont}/img_%02i.jpg" %i, blur)   
        i += 1
    cont += 1