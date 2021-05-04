from PIL import ImageFont, ImageDraw, Image
import cv2
import time
import numpy as np
import skimage.measure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import glob

clf = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial', max_iter=100000, verbose=1, tol=0.00001)

X = []
Y = [] 

# 1 - Através de um for salvar todas as imagens como X e seus rótulos em Y
im_folder = os.listdir('ocr-treino/num_cortes_bin/')

#Cria uma imagem com cada caracter e número
for c in im_folder: 
    for n in glob.glob(f"ocr-treino/num_cortes_bin/{c}/*.jpg"):
        img = cv2.imread(n)
        #normaliza [0,1] a imagem
        img = img.astype(float) / 255.0
        #adiciona imagem em x
        X.append(img)
        #atribui em Y a classe da imagem com ASCII
        Y.append(str(c))

#Converte X em Array e aplica Reshape em X para o formato correto para treinamento
X = np.array(X).reshape(len(Y),-1)

#converte Y em Array
Y = np.array(Y)
Y = Y.reshape(-1)

#cria o objeto para escala dos vetores
ss = StandardScaler()
#print("treinando modelo")

#Normaliza [-1,1] as imagens em X
X = X - (X/127.5)

#Treina o modelo
clf.fit(X, Y)

#aplica escala nos vetores
ss.fit(X)
#print("salvando modelo")

#Salva modelo
joblib.dump((clf, ss), "ocr-treino/caracteres_num_cnh.pkl", compress=3)
#print("modelo salvo")

#aliação de métricas

'''kf = KFold(n_splits=10, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    reg = LogisticRegression(max_iter=5000).fit(X_train, y_train)
    pre = reg.predict(X_test)
    print(reg.score(X_test, y_test))'''
