import cv2
import re
import pytesseract
from pytesseract import Output

######################################################################################
# Obtendo caixas delimitadoras
img = cv2.imread('processada.jpg')
d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 40:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#cv2.imshow('img', img)
#cv2.waitKey(0)
########################################################################################

########################################################################################
# Obtendo caixas delimitadoras apenas ao redor de números
img1 = cv2.imread('processada.jpg')
d = pytesseract.image_to_data(img1, output_type=Output.DICT)
keys = list(d.keys())

date_pattern = '^[0-9]*$' #mudar parâmetros para conseguir pegar CPF e datas

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        if re.match(date_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('imgJustDig', img1)
cv2.waitKey(0)
#########################################################################################