import cv2
import numpy as np
import all_functions as fuck
import time
#import color_recognize

cap=cv2.VideoCapture(0)
while(True):
    time.sleep(2)
    ret, img = cap.read()
    if (ret == False):
        break
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.imshow('resize',res_img)
    # take ROI
    r = cv2.selectROI(img)  # return x,y,w,h
    np.save('roi', r)
    arena = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    arena = cv2.resize(arena, (720, 720))
    cv2.imshow('arena', arena)
    cv2.imwrite('./arena.jpg', arena)
    # taking roi for colour detect
    n = 5
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
    LRR, URR = fuck.clr_range(LRR, URR, arena)
    print(LRR)
    print(URR)
    np.save('lrr', LRR)
    np.save('urr', URR)
    # range for lower and upper yellow
    LRY, URY = fuck.clr_range(LRY, URY, arena)
    print(LRY)
    print(URY)
    np.save('lry', LRY)
    np.save('ury', URY)
    # range for lower and upper blue
    LRB, URB = fuck.clr_range(LRB, URB, arena)
    print(LRB)
    print(URB)
    np.save('lrb', LRB)
    np.save('urb', URB)
    # range for lower and upper green
    LRG, URG = fuck.clr_range(LRG, URG, arena)
    print(LRG)
    print(URG)
    np.save('lrg', LRG)
    np.save('urg', URG)
    # range for lower and upper white
    LRW, URW = fuck.clr_range(LRW, URW, arena)
    print(LRW)
    print(URW)
    np.save('lrw', LRW)
    np.save('urw', URW)
    low = [LRR, LRY, LRB, LRW, LRG]
    high = [URR, URY, URB, URW, URG]

    # taking each square box separately
    leng = arena.shape
    row = leng[0] // n
    col = leng[1] // n
    print(row)
    print(col)

    # numpy array declaration
    shape_mat = np.zeros((n, n), dtype=np.int16)  # for shape
    edge_mat = np.zeros((n, n), dtype=np.int16)  # for no of corners
    color_mat = np.zeros((n, n), dtype=np.int16)  # for color
    for i in range(n):
        for j in range(n):
            shape_mat[i][j] = -1
    # color recognise and shape detect
    fuck.recog_col(color_mat, shape_mat, row, col, LRR, URR, arena, 0)
    fuck.recog_col(color_mat, shape_mat, row, col, LRY, URY, arena, 1)
    fuck.recog_col(color_mat, shape_mat, row, col, LRW, URW, arena, 5)
    fuck.recog_col(color_mat, shape_mat, row, col, LRB, URB, arena, 2)
    fuck.recog_col(color_mat, shape_mat, row, col, LRG, URG, arena, 4)
    np.transpose(color_mat)
    print(color_mat)
    # print(edge_mat)
    print(shape_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()