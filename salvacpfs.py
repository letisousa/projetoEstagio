import glob
import numpy as np
import cv2
import imutils
import pytesseract
from pytesseract import Output
import re
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (11,7)

def sharpenFunction(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #matriz de convolução que aplica nitidez
    return cv2.filter2D(image, -1, kernel)

def filterFunction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T,Thresh1) = cv2.threshold(gray, 187, 255, cv2.THRESH_TRUNC)
    Thresh2 = cv2.adaptiveThreshold(Thresh1, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 231,50)
    return Thresh2

def settozero(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO)
    return thresh

images = glob.glob("cpfs/*.jpg")
i = 0

for p in images:
    
    img = cv2.imread(p)
    cv2.imshow('Imagem original', img)
    k = cv2.waitKey(60)
    if k == ord('q'):
        exit()

    #tozero = settozero(img)
    #ret,thresh = cv2.threshold(tozero,125,255,cv2.THRESH_BINARY)
    nitida = sharpenFunction(img)
    filt = filterFunction(nitida)
    blur = cv2.medianBlur(filt, 3)
    cv2.imwrite("ocr-treino/cpf_proc/cpf_proc%02i.jpg" %i, blur)
    i += 1

    cv2.imshow('CPF', blur)
    k = cv2.waitKey(60)
    if k == ord('q'):
        exit()