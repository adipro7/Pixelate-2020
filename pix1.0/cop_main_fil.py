import cv2
import numpy as np
import clr_det
import color_recognize

# cap=cv2.VideoCapture(0)
# while(true)
#   ret,img=cap.read()

img = cv2.imread('C:/Users/DELL/Pictures/Screenshots/Screenshot (244).png', 1)
# cv2.imshow('image',img)
siz = img.shape
rat = siz[0] / siz[1]
wei = 1500
hei = rat * wei
print(wei)
print(hei)

print(siz)

res_img = cv2.resize(img, (int(wei), int(hei)))

# cv2.imshow('resize',res_img)
# take ROI
r = cv2.selectROI(res_img)# return x,y,w,h
arena = res_img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
arena = cv2.resize(arena, (720, 720))
cv2.imshow('arena', arena)

# taking roi for colour detect

LRR = [255, 255, 255]
URR = [0, 0, 0]
LRG = [255, 255, 255]
URG = [0, 0, 0]
LRB = [255, 255, 255]
URB = [0, 0, 0]
LRY = [255, 255, 255]
URY = [0, 0, 0]
LRW = [255, 255, 255]
URW = [0, 0, 0]

# range for lower and upper red
LRR, URR = clr_det.clr_range(LRR, URR, arena)
print(LRR)
print(URR)

# range for lower and upper yellow
LRY, URY = clr_det.clr_range(LRY, URY, arena)
print(LRY)
print(URY)

# range for lower and upper blue
LRB, URB = clr_det.clr_range(LRB, URB, arena)
print(LRB)
print(URB)

# range for lower and upper green
LRG, URG = clr_det.clr_range(LRG, URG, arena)
print(LRG)
print(URG)

# range for lower and upper white
LRW, URW = clr_det.clr_range(LRW, URW, arena)
print(LRW)
print(URW)

low = [LRR, LRY, LRB, LRW, LRG]
high = [URR, URY, URB, URW, URG]

# taking each square box separately
leng = arena.shape
row = leng[0] // 9
col = leng[1] // 9
print(row)
print(col)

# numpy array declaration
shape_mat = np.zeros((9, 9), dtype=np.int16)  # for shape
edge_mat = np.zeros((9, 9), dtype=np.int16)  # for no of corners
color_mat = np.zeros((9, 9), dtype=np.int16)  # for color
for i in range(9):
    for j in range(9):
        shape_mat[i][j]=-1
# color recognise and shape detect
color_recognize.recog_col(color_mat,shape_mat, row, col, LRR, URR, arena, 0)
color_recognize.recog_col(color_mat,shape_mat, row, col, LRY, URY, arena, 1)
color_recognize.recog_col(color_mat,shape_mat, row, col, LRW, URW, arena, 5)
color_recognize.recog_col(color_mat,shape_mat, row, col, LRB, URB, arena, 2)
color_recognize.recog_col(color_mat,shape_mat, row, col, LRG, URG, arena, 4)
np.transpose(color_mat)
print(color_mat)
#print(edge_mat)
print(shape_mat)
cv2.waitKey(0)
cv2.destroyAllWindows()