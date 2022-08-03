import requests
import cv2
import numpy as np
import os
import imutils
import idTag

passtoTable = '.\assets\table.jpg'

class tagDetector:
    def __init__(self, img):
        self.table = cv2.imread(passtoTable,cv2.IMREAD_REDUCED_COLOR_8)
        self.table = self.rescaleFrame(self.table,0.4)
        self.cornerTabl, self.idTabl = self.findArucoMarkers(self.table)

        self.image = img
        self.arrayCorner, self.arrayId = self.findArucoMarkers(self.image)

        self.centerPos = [0]*2
        self.arrayPos=self.getPos(self.arrayCorner)
        
        self.H=None
        self.Hcalculated=False
        self.imgH=None
        


    #Renvoie un tableau contenant les coins et un autre tableau contenant
    # les ID des tags détectés
    def findArucoMarkers(self,img, markerSize = 4, totalMarkers=100, draw=True):
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

 

    #Renvoie un tableau contenant les coordonées du centre du tag Aruco
    def getPos(self,bbox):
        tabPos = [[0 for x in range(2)] for y in range(len(bbox))] 
        for i in range(len(tabPos)):
            #Calcul le x moyen
            tabPos[i][0]=bbox[i][0][0][0]+bbox[i][0][1][0]+bbox[i][0][2][0]+bbox[i][0][3][0]
            tabPos[i][0]=(int)(tabPos[i][0]/4) - self.centerPos[0]

            #Calcul le y moyen
            tabPos[i][1]=bbox[i][0][0][1]+bbox[i][0][1][1]+bbox[i][0][2][1]+bbox[i][0][3][1]
            tabPos[i][1]=(int)(tabPos[i][1]/4) - self.centerPos[1]
        
        print(tabPos)
        print('fin')
        return tabPos

    #Renvoie la matrice d'homography
    def get_matrixH(self,cornerTabl, cornerCam):
        cam_pts = np.array(cornerCam).reshape(-1,1,2)
        table_pts = np.array(cornerTabl).reshape(-1,1,2)
        H, mask = cv2.findHomography(cam_pts, table_pts, cv2.RANSAC,5.0)
        return H

    #Renvoie l'image de la caméra projetée sur la table de jeu
    def get_plan_view(self,cam, table, H):
        plan_view = cv2.warpPerspective(cam, H, (table.shape[1], table.shape[0]))
        return plan_view

    #Rescale l'image 
    def rescaleFrame(self,frame, scale=0.75):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width,height)
        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    
    def refresh(self,img):

        #Si la matrice d'homography est calculer
        if (self.Hcalculated):
            self.image = img
            self.imgH = self.get_plan_view(self.image, self.table, self.H)
            self.arrayCorner, self.arrayId = self.findArucoMarkers(self.imgH)
            
        else : 
            #Si des tags sont détectés
            if self.arrayId is not None :
                for i in range(len(self.arrayId)):
                    #Si on trouve le tag du centre
                    if self.arrayId[i] == idTag.CENTRE:
                                                       
                            print(self.arrayCorner)
                            #On réalise l'Homography
                            self.H = self.get_matrixH(self.cornerTabl, self.arrayCorner[i]) 
                            self.Hcalculated = True 
                            self.imgH=self.get_plan_view(self.image, self.table, self.H)
                            self.arrayCorner, self.arrayId = self.findArucoMarkers(self.imgH)
                            
                            self.centerPos[0]=self.arrayPos[i][0]
                            self.centerPos[1]=self.arrayPos[i][1]

                            break

            #Si H n'a pas été calculé on refresh avec l'image classique 
            if not self.Hcalculated:
                self.image = img
                self.arrayCorner, self.arrayId = self.findArucoMarkers(self.image)
                
        
        self.arrayPos = self.getPos(self.arrayCorner)

    

