from PIL import ImageFont, ImageDraw, Image
import cv2
import time
import numpy as np
import skimage.measure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

clf = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial', max_iter=100000, verbose=1, tol=0.00001)

X = []
Y = [] 

#Adiciona uma imagem vazia em X e atribui a classe "fundo" em Y para esta imagem
img = np.zeros((50,30,3),np.uint8)
X.append(img)
Y.append("fundo")

#carrega o diretório com as fontes
fonts = os.listdir('ocr-treino/fonts/')

#carrega o diretório onde serão gravadas as imagens das fontes
im_folder = os.listdir('ocr-treino/images/')


#cria uma pasta vazia para cada letra e número
if len(im_folder) < 1:
    for c in range(48,91):
        print("aqui 2")
        if c < 58 or c > 64:
            print("aqui 3")
            os.system("mkdir ocr-treino/images/" + str(chr(c)))
            print("aqui 4")
            
im_folder = os.listdir('ocr-treino/images/')

print(sorted(im_folder))