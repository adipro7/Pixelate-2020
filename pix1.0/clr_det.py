import cv2
import numpy as np
# finding color range RYBGW
def clr_range(LR,UR,arena):
    for i in range(1):
        r = cv2.selectROI(arena)
        col_img = arena[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        row, col, hei = col_img.shape
        for r1 in range(row):
            for c in range(col):
                pixe = col_img[r1][c]
                for k in range(3):
                    LR[k] = min(LR[k], max(pixe[k]-15,0))
                    UR[k] = max(UR[k], min(pixe[k]+15,255))
    return LR,UR