import cv2
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image
import os
import random
import glob

def settozero(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO)
    return thresh

clf,pp = joblib.load('ocr-treino/caracteres.pkl') 
bi = 5
lista_imagens = glob.glob('cpfs/*jpg')

font = cv2.FONT_HERSHEY_SIMPLEX
ratio = .9  # resize ratio

for c in lista_imagens:
    
    tess = []
    img = cv2.imread(c)

    black = np.zeros_like(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    foi = 0
    ncars  = 0
    x_letter = []
    letter = []
    lit = []
    res = ''

    img2 = np.ones_like(gray)
    tozero = settozero(img)
    ret,thresh = cv2.threshold(tozero,125,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for (i, contour) in enumerate(contours):
        xx,yy,ww,hh = cv2.boundingRect(contour)
        #print("x:", xx, "y:", yy, "w:", ww, "h:", hh)
        if ww > 10 and ww < 100 and hh > 10 and xx > 0 and yy > 0:
            th = thresh.copy()
            roi = th[yy-bi:yy+hh+bi+25,xx-bi:xx+ww+bi+4] 
            # cv2.imshow('roi'+str(i),roi)
            # k = cv2.waitKey(60)
            # if k == ord('q'):
            #     exit()
            roi2 = th[yy-bi-20:yy+hh+bi+25,xx-bi-20:xx+ww+bi+14]
            if roi.shape[0] > 0: 
                roi = cv2.resize(roi, (30,50)) 
                F = np.array(roi).reshape(1,-1)
                pre = clf.predict(F)
                score = clf.predict_proba(F)
                print("Letra - {}  ", pre)
                print("Score - {}  ", np.argmax(score))

                msg = str(pre).replace("_"," ").replace("[","").replace("]","").replace("'","")

                x_letter.append(xx)
                letter.append(msg) 

    x_c = []
    if len(x_letter) > 1:
        for it in sorted(x_letter):
            x_c.append(it) 
            ll = x_letter.index(it)
            lit.append(ll)
        for sss in lit:
            res = res + str(letter[sss])
        res = list(res[0:14])

        print(res)

        res = str(res).replace("_","").replace("[","").replace("]","").replace("'","").replace(",","")


