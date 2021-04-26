import cv2 
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image
import os
import random
import glob 

clf,pp = joblib.load('ocr-treino/caracteres.pkl') 
bi = 5

#criando lista com as imagens dos cpfs
imagens = glob.glob('ocr-treino/cpf_proc/*jpg')

while True:
    idx = random.randint(0, len(imagens)-1)
    img = cv2.imread(imagens[idx])
    cv2.imshow('CPF', img)
    k = cv2.waitKey(60)
    if k == ord('q'):
       exit()

