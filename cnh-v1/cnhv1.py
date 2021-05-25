import cv2
import glob
import os
import numpy as np
import joblib

#################################################
#     CAPTURA FOTO E SALVA NO DIRETÓRIO         #

'''def main(args):

    winName = 'Pressione S para capturar uma imagem'
    
    cam_port = 1
    camera = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 4096) #4096
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) #2160
    
    emLoop = True

    list_images = os.listdir("cnhs/")
    i = len(list_images)

    while(emLoop):
        
        ret, img = camera.read()
        imgS= cv2.resize(img, (960,540))
        cv2.imshow(winName, imgS)
        
        k = cv2.waitKey(1)
        if k == 27:
            emLoop = False
        if k == ord('s'):
            cv2.imwrite('cnhs/imagem%02i.jpg' %i, img)
            emLoop = False
            i+=1
        
    cv2.destroyAllWindows()
    camera.release()
    
    return 0

if __name__ == '__main__':
    
    import sys
    sys.exit(main(sys.argv))'''

#               FIM DA CAPTURA                  #
################################################# 


def settozero(img): #filtro das imagens
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO)
    return thresh

def cropRoiCpf(img):
    imgcrop = img[1270:1325, 1673:1922] 
    return imgcrop

def cropRoiNReg(img):
    imgcrop = img[1555:1615, 1400:1630]
    return imgcrop

def cropRoiNasc(img):
    imgcrop = img[1267:1325, 1905:2091]
    return imgcrop

def cropRoiVal(img):
    imgcrop = img[1540:1620, 1660:1850]
    return imgcrop

clf_cpf, pp_cpf = joblib.load('caracteres_num_cnh.pkl') #carrega arquivo treinado para reconhecer caracteres numéricos
clf_data,pp_data = joblib.load('caracteres_data_cnh.pkl')
bi = 5 #valor usado para calcular "bordas" das rois
lista_imagens = glob.glob('cnhs/*.jpg') #lista que contém as fotos das cnhs

def getCpf(img):

    roicpf = cropRoiCpf(img)

    x_letter = []
    letter = [] 
    lit = []
    cpf = ''
    
    tozero = settozero(roicpf)
    ret,thresh = cv2.threshold(tozero,125,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for (i, contour) in enumerate(contours):
        xx,yy,ww,hh = cv2.boundingRect(contour)
        if ww > 10 and ww < 100 and hh > 14:
            th = thresh.copy()
            roi = th[yy-bi:yy+hh+bi+25,xx-3:xx+ww+3] 
            if roi.shape[0] > 0 and roi.shape[1] > 0: 
                roi = cv2.resize(roi, (30,50))
                F = np.array(roi).reshape(1,-1)
                pre = clf_cpf.predict(F)

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
            cpf = cpf + str(letter[sss])
        cpf = list(cpf[0:14])
        global newcpf
        newcpf = []
        count = 0
        for n in cpf:
            if count < 11:
                newcpf.append(n)
                count += 1       
        newcpf = str(newcpf).replace("_","").replace("[","").replace("]","").replace("'","").replace(",","")

    return newcpf

def getNReg(img):
    
    roiNReg = cropRoiNReg(img)

    x_letter = []
    letter = [] 
    lit = []
    reg = ''

    list_reg = os.listdir("verRois/")
    j = len(list_reg) 

    tozero = settozero(roiNReg)
    ret,thresh = cv2.threshold(tozero,125,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for (i, contour) in enumerate(contours):
        xx,yy,ww,hh = cv2.boundingRect(contour)
        if ww > 8 and ww < 18 and hh > 14:
            #print("x:", xx, "y:", yy, "w:", ww, "h:", hh)
            th = thresh.copy()
            roi = th[yy-bi:yy+hh+bi+25,xx-3:xx+ww+3] 
            if roi.shape[0] > 0 and roi.shape[1] > 0: 
                '''cv2.imshow('roi'+str(i), roi)
                k = cv2.waitKey(60)
                if k == ord('q'):
                    exit()'''
                roi = cv2.resize(roi, (30,50))
                cv2.imwrite(f"verRois/%07i.jpg" %j, roi)
                j += 1 
                F = np.array(roi).reshape(1,-1)
                pre = clf_cpf.predict(F)

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
            reg = reg + str(letter[sss])
        reg = list(reg[0:14])
        global newreg
        newreg = []
        count = 0
        for n in reg:
            if count < 11:
                newreg.append(n)
                count += 1       
        reg = str(newreg).replace("_","").replace("[","").replace("]","").replace("'","").replace(",","").replace("b", "/")

    return reg

def getNasc(img):
    
    roiNasc = cropRoiNasc(img)
    
    x_letter = []
    letter = []
    lit = []
    nasc = ''

    tozero = settozero(roiNasc)
    ret,thresh = cv2.threshold(tozero,125,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for (i, contour) in enumerate(contours):
        xx,yy,ww,hh = cv2.boundingRect(contour)
        #print("x:", xx, "y:", yy, "w:", ww, "h:", hh)
        if ww > 10 and ww < 100 and hh > 10 and xx > 7 and yy > 12:
            th = thresh.copy()
            roi = th[yy-bi:yy+hh+bi+25,xx-3:xx+ww+3]
            if roi.shape[0] > 0 and roi.shape[1] > 0: 
                cv2.imshow('roi'+str(i), roi)
                k = cv2.waitKey(60)
                if k == ord('q'):
                    exit()
                roi = cv2.resize(roi, (30,50))
                F = np.array(roi).reshape(1,-1)
                pre = clf_data.predict(F)

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
            nasc = nasc + str(letter[sss])
        nasc = list(nasc[0:11])
        global newnasc
        newnasc = []

        for c in nasc:
            if c != 'b':
                newnasc.append(c)

        nasc = str(nasc).replace("_","").replace("[","").replace("]","").replace("'","").replace(",","").replace("b", "")

    return nasc

def getVal(img):
    
    roiNasc = cropRoiNasc(img)
    
    x_letter = []
    letter = []
    lit = []
    nasc = ''
    list_reg = os.listdir("verRois/")
    j = len(list_reg)

    tozero = settozero(roiNasc)
    ret,thresh = cv2.threshold(tozero,125,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for (i, contour) in enumerate(contours):
        xx,yy,ww,hh = cv2.boundingRect(contour)
        #print("x:", xx, "y:", yy, "w:", ww, "h:", hh)
        if ww > 10 and ww < 100 and hh > 10 and xx > 7 and yy > 12:
            th = thresh.copy()
            roi = th[yy-bi:yy+hh+bi+25,xx-3:xx+ww+3]
            if roi.shape[0] > 0 and roi.shape[1] > 0: 
                cv2.imshow('roi'+str(i), roi)
                k = cv2.waitKey(60)
                if k == ord('q'):
                    exit()
                roi = cv2.resize(roi, (30,50))
                cv2.imwrite(f"verRois/%05i.jpg" %j, roi)
                j += 1
                F = np.array(roi).reshape(1,-1)
                pre = clf_data.predict(F)

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
            nasc = nasc + str(letter[sss])
        nasc = list(nasc[0:11])
        global newnasc
        newnasc = []

        for c in nasc:
            if c != 'b':
                newnasc.append(c)

        nasc = str(nasc).replace("_","").replace("[","").replace("]","").replace("'","").replace(",","").replace("b", "")

    return nasc

for c in lista_imagens:
        
    img = cv2.imread(c) #lê imagens
    #cpf = getCpf(img)
    #print('CPF:', cpf)
    nreg = getNReg(img)
    print('Reg:', nreg)
    #nasc = getNasc(img)
    #print('Nasc:', nasc)
