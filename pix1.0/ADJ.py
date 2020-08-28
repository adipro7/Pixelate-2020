import numpy as np
import cv2
import cv2.aruco as aruco
image=cv2.imread('C:/Users/DELL/Pictures/ar1.png')
'''Here you will be required to check the Aruco type whether it is
 4X4, 5X5, 6X6, 7X7 and change accordingly'''
cv2.imshow('w',image)
cv2.waitKey(0)
for i in range(4):
    if(i>=1):
        image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    aruco_dict=aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters=aruco.DetectorParameters_create()
    corners,ids,rejectedImgPoints=aruco.detectMarkers(image,aruco_dict,parameters=parameters)
    print("corners")
    print(corners)
    '''detection of center of Aruco. corners is a list having coordinates of 4 corners. 
    Taking half of x of 1st and 2nd and half of y of 2nd and third.'''
    #for cor in corners:
        #print(cor)
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
        #print(xc,yc)
    aruco.drawDetectedMarkers(image,corners,ids) #This function draws the boundary of the aruco
    cv2.imshow('aruco',image)
    cv2.waitKey(0)
#cv2.imwrite('detected_aruco.jpg',image)
cv2.waitKey(0)
#IDENTIFY THE CELL YOURSELF
#contributed by: APG