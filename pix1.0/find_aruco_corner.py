import numpy as np
import cv2
import cv2.aruco as aruco


cap = cv2.VideoCapture(0)
while (True):
    ret, image = cap.read()
    if(ret==False):
        break       
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break
    '''Here you will be required to check the Aruco type whether it is
     4X4, 5X5, 6X6, 7X7 and change accordingly'''
    aruco_dict=aruco.Dictionary_get(aruco.DICT_6X6_1000)
    parameters=aruco.DetectorParameters_create()
    corners,ids,rejectedImgPoints=aruco.detectMarkers(image,aruco_dict,parameters=parameters)
    #print(type(corners))
    #print(len(corners))
    #print(corners[0][0][0][0],corners[0][0][0][1])
    #cv2.imshow('imag',img)
    #cv2.waitKey(0)'''

    '''detection of center of Aruco. corners is a list having coordinates of 4 corners. 
    Taking half of x of 1st and 2nd and half of y of 2nd and third.'''
    #print(type(ids))
    if ids is None:

        continue
    img=cv2.circle(image,(corners[0][0][0][0],corners[0][0][0][1]),5,(0,0,255),2)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    cv2.destroyWindow('img')
    print("corners")
    print(corners)
    for x in range(0,ids.shape[0]):
        p1 = max(corners[x][0][0][0], corners[x][0][1][0],
                corners[x][0][2][0], corners[x][0][3][0])
        p2 = min(corners[x][0][0][0], corners[x][0][1][0],
                 corners[x][0][2][0], corners[x][0][3][0])
        q1 = max(corners[x][0][0][1], corners[x][0][1][1],
                 corners[x][0][2][1], corners[x][0][3][1])
        q2 = min(corners[x][0][0][1], corners[x][0][1][1],
                 corners[x][0][2][1], corners[x][0][3][1])
        xc = int(p2+abs(p1-p2)/2)
        yc = int(q2+abs(q1-q2)/2)
        #A small dot is shown at center. See
        image[yc][xc]=(0,255,0)
        print(xc,yc)
        aruco.drawDetectedMarkers(image,corners,ids) #This function draws the boundary of the aruco
        #cv2.imshow('aruco',image)
        #cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
