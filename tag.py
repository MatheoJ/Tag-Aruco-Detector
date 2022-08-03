import requests
import cv2
import numpy as np
import os
import imutils
#url = "http://192.168.1.249:8080/shot.jpg"

from enum import Enum
class tagEnum(Enum):
    BLEU = 13
    VERT = 36
    ROUGE = 47
    CENTRE = 42
    ROCHER = 17 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")



#Renvoie un tableau contenant les coins et un autre tableau contenant
# les ID des tag détectées
def findArucoMarkers(img, markerSize = 4, totalMarkers=100, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.Dictionary_get(key)
    arucoParam = cv2.aruco.DetectorParameters_create()
    bboxs, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
        
    if bboxs:
        len(bboxs)
        #printPos(bboxs)
        print(bboxs[0][0])
        #print('fin')
        print(ids)

    if draw:
        cv2.aruco.drawDetectedMarkers(img, bboxs) 

    return bboxs, ids

def printDist(bbox,markerSize =4):
    tl = bbox[0][0][0],bbox[0][0][1]
    tr = bbox[0][1][0],bbox[0][0][1]
    print(tl[0])
    print(tr[0]) 


#Renvoie un tableau contenant les coordonées du centre du tag Aruco
def getPos(bbox,markerSize =4):
    tabPos = [[0 for x in range(2)] for y in range(len(bbox))] 
    for i in range(len(tabPos)):
        #Calcul le x moyen
        tabPos[i][0]=bbox[i][0][0][0]+bbox[i][0][1][0]+bbox[i][0][2][0]+bbox[i][0][3][0]
        tabPos[i][0]=(int)(tabPos[i][0]/4)

        #Calcul le y moyen
        tabPos[i][1]=bbox[i][0][0][1]+bbox[i][0][1][1]+bbox[i][0][2][1]+bbox[i][0][3][1]
        tabPos[i][1]=(int)(tabPos[i][1]/4)
    
    print(tabPos)
    print('fin')
    return tabPos

#Renvoie la matrice d'homography
def get_matrixH(cornerTabl, cornerCam):
    cam_pts = np.array(cornerCam).reshape(-1,1,2)
    table_pts = np.array(cornerTabl).reshape(-1,1,2)
    H, mask = cv2.findHomography(cam_pts, table_pts, cv2.RANSAC,5.0)
    #print("H:")
    #print(H)
    return H

#Renvoie l'image de la caméra projetée sur la table de jeu
def get_plan_view(cam, table, H):

    plan_view = cv2.warpPerspective(cam, H, (table.shape[1], table.shape[0]))
    return plan_view

#Rescale l'image 
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


table = cv2.imread(r'G:\Mon Drive\Perso\python\ARUCO\Photos\table.jpg',cv2.IMREAD_REDUCED_COLOR_8)
table = rescaleFrame(table,0.3)
cornerTabl, idTabl = findArucoMarkers(table)
cv2.imshow('table', table)

Hcalculated = False

while True:
    sucess, img = cap.read()
    """ img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)"""
    cornerCam, idCam = findArucoMarkers(img) 

    if (not Hcalculated) and (idCam is not None) :
        for i in range(len(idCam)):
            if idCam[i] == 42:
                    print(cornerCam[0][i])
                    H = get_matrixH(cornerTabl, cornerCam[0][i]) 
                    Hcalculated = True      
    if (Hcalculated):
        plan_view=get_plan_view(img, table, H)
        cv2.imshow('webcam', plan_view)
    else :
        cv2.imshow('webcam', img)

    if cv2.waitKey(1) == 27: 
            break
    
cap.release()
cv2.destroyAllWindows()