import cv2
import glob
import numpy as np
import os

def sift_align(img1, img2):
    
    # Converte imagens para cinza
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detecta recursos ORB
    sift = cv2.SIFT_create(5000)
    keypoints1, descriptors1 = sift.detectAndCompute(img1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2Gray, None)

    # Match features (Recursos de combinações)
    matcher = cv2.DescriptorMatcher_create(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Organiza os matches por pontuação
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remova combinações não tão boas
    numGoodMatches = int(len(matches) * 2)
    matches = matches[:numGoodMatches]

    # salva os melhores matches
    imgMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imgMatches)

    # Extrai a localizão do bons matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Encontra homografia
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Usa homografia
    height, width, channels = img2.shape
    
    img1Reg = cv2.warpPerspective(img1, h, (width, height))

    return img1Reg, h


imgsbr = glob.glob("alinharbr/*.jpeg")
imgspb = glob.glob("alinharpb/*.jpeg")
imgRefbr = cv2.imread("alinharbr/refvazia.jpg")
imgRefpb = cv2.imread("alinharpb/imagem20.jpg")

i = 0

# for c in imgsbr:

#     img = cv2.imread(c) 
#     realinhada, h = sift_align(img, imgRefbr)
    
#     list_reg = os.listdir("alinhadas/")
#     i = len(list_reg)
    
#     cv2.imwrite('alinhadas/alinhada%02i.jpg' %i, realinhada)

for d in imgspb:

    img1 = cv2.imread(d)
    realign, hh = sift_align(img1, imgRefpb)

    list_reg1 = os.listdir("alinhadas/")
    j = len(list_reg1)
    cv2.imwrite('alinhadas/alinhada%02i.jpg' %j, realign) 