import cv2
import numpy as np

def recog_col(color_mat,shape_mat,row,col,LRW,URW,arena,num):
    lrw = np.array([LRW])
    urw = np.array([URW])
    mask = cv2.inRange(arena, lrw, urw)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area_of_cnt = cv2.contourArea(cnt)
        if (area_of_cnt>20):
            cen = cv2.moments(cnt)
            cenx = int(cen["m10"] / cen["m00"])
            ceny = int(cen["m01"] / cen["m00"])
            cenx = cenx // row
            ceny = ceny // col
            color_mat[ceny][cenx] = num
            # shape recog
            if(num<2):
                rect = cv2.minAreaRect(cnt)# making min_area_rect aroung contours
                area_of_rect=rect[1][0]*rect[1][1]# area of contours
                box = cv2.boxPoints(rect)# recovering 4 point of min_rect
                box = np.int0(box)
                cv2.drawContours(mask, [box], 0, (100,100,255), 2)# drawing rectangle around contours
                cv2.imshow('area_of',mask)
                cv2.waitKey(100)
                rat=area_of_cnt/area_of_rect# taking ratio of (area of conotur/area of rectangle)
                if rat>=0.87:
                    print(rat,1)
                    shape_mat[ceny][cenx]=1
                else:
                    print(rat,0)
                    shape_mat[ceny][cenx]=0