import requests
import cv2
import numpy as np
import os
import imutils
import idTag 
from tagDetector import tagDetector

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#Permet de récupérer une image en fonction du flux souhaiter
def getStream(type):
    if type=="webcam":
        sucess, img = cap.read() 
    elif type=="phone":
        url = "https://10.141.211.64:8080/shot.jpg"
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)
    return(img)



img = getStream("webcam") 
camTower=tagDetector(img)


passToPhoto = r'G:\Mon Drive\Perso\python\ARUCO\Photos\photo.jpg'
cv2.imread(passToPhoto)


while True:
    img = cv2.imread(passToPhoto) 
    camTower.refresh(img)


    if (camTower.Hcalculated):
        cv2.imshow('webcam',camTower.imgH)
    else :
        cv2.imshow('webcam',camTower.image)

    if cv2.waitKey(1) == 27: 
            break


    
cap.release()
cv2.destroyAllWindows()
 
