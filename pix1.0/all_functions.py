import cv2
import numpy as np
import math
import time

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
                #cv2.imshow('area_of',mask)
                #cv2.waitKey(100)
                rat=area_of_cnt/area_of_rect# taking ratio of (area of conotur/area of rectangle)
                if rat>=0.87:
                    print(rat,1)
                    shape_mat[ceny][cenx]=1
                else:
                    print(rat,0)
                    shape_mat[ceny][cenx]=0

def return_cord(cell_id,cell_num,n=9):
    for i in range(n):
        for j in range(n):
            if(cell_num[i][j]==cell_id):
                return (i,j)

# for update of weights of which is decided by color and shape of weapons
def update_weight(color_mat,shape_mat,cell_adj,cell_num,var_cell_wei,shape,color,n=9):
    u_range=n*n+1 # 82 in case of n
    for i in range(1,u_range,1):
        for num in range(cell_adj[i]):
            if(num==0):
                continue
            else:
                r,c=return_cord(num,cell_num)
                if(color_mat[r][c]==color & shape_mat[r][c]==shape):
                    var_cell_wei.append(0) # putting 0 for same color and shape
                else:
                    var_cell_wei.append(1) #putting 1 for different shape or color

# to connect a cell to its four neighbour whichever exists
def connect_edges(cell_adj,cell_wei,cell_num,cell_id,n=9):
    r, c = return_cord(cell_id,cell_num)
    upx = r - 1
    dwx = r + 1
    ly = c - 1
    ry = c + 1
    if (upx >= 0):
        cell_adj[cell_id].append(cell_num[upx][c])
        cell_wei[cell_id].append(1)
    if (dwx < n):
        cell_adj[cell_id].append(cell_num[dwx][c])
        cell_wei[cell_id].append(1)
    if (ly >= 0):
        cell_adj[cell_id].append(cell_num[r][ly])
        cell_wei[cell_id].append(1)
    if (ry < n):
        cell_adj[cell_id].append(cell_num[r][ry])
        cell_wei[cell_id].append(1)

# for remove edges of horcurex when weapons reaches to Horcurex
def remove_edges(cell_wei,cell_adj,cell_id):
    if cell_adj[cell_id] is None:
        return
    while(len(cell_adj[cell_id])>1):
        cell_adj[cell_id].pop()
        cell_wei[cell_id].pop()

# update color and shape matrix again
def update_color_shape(Horcruxes,Weapons,unvis_horcurex,arena,shape_mat,color_mat,row,col,LRR,URR,LRY,URY,up_date):
    new_color_mat=np.full((9,9),-1)
    new_shape_mat=np.full((9,9),-1)
    recog_col(new_color_mat, new_shape_mat, row, col, LRR, URR, arena, 0)
    recog_col(new_color_mat, new_shape_mat, row, col, LRY, URY, arena, 1)
    if(up_date==0):
        for ele in range (Horcruxes):
            if(new_shape_mat[r][c]!=-1):
                r,c=return_cord(ele)
                color_mat[r][c]=new_color_mat[r][c]
                shape_mat[r][c]=new_shape_mat[r][c]
    else:
        for ele in (Weapons):
            r,c=return_cord(ele)
            if(new_shape_mat[r][c]!=-1):
                shape_mat[r][c]=new_shape_mat[r][c]
                color_mat[r][c]=new_color_mat[r][c]

# finding centres of each cells of arena
def find_centre(arena,n=9):
    cord=[]
    l,b=arena.shape
    row=l/n
    col=b/n
    for i in range(n):
        for j in range(n):
            cord.append(col*(i+(1/2)),row*(j+(1/2)))
    return cord

# return front direction of boat
def bot_vector(p,i,j):
    return p[0][0][i]-p[0][0][j]

# return direction in which boat have to move
def dirn_of_mov_vector(boat_centre,next_cell,r):
    cell_cent=find_centre(next_cell)
    boat_centre[0]-=r[0]
    boat_centre[1]-=r[1]
    return cell_cent-boat_centre

# return distance between boat_centre and cell_cent
def find_dis(boat_centre,next_cell,r):
    cell_cent = find_centre(next_cell)
    boat_centre[0]-= r[0]
    boat_centre[1]-= r[1]
    return math.sqrt(((cell_cent[0]-boat_centre[0])**2)+((cell_cent[1]-boat_centre[1])**2))

# determine angle between boat direction and direction of movemnet
def cross_pro(dirn_of_mov=[1,1],boat_vector=[0,1]):
    a=np.array(dirn_of_mov)
    b=np.array(boat_vector)
    print(np.cross(a, b))
    mag = (math.sqrt(dirn_of_mov[0] ** 2 + dirn_of_mov[1] ** 2)) * (math.sqrt(boat_vector[0] ** 2 + boat_vector[1] ** 2))
    print(math.degrees(math.asin(np.cross(a,b)/mag)))

def bot_movement(go_path,n=9,cord):
    dis = 10000
    flag2=False
    for box in (go_path):
        if(box==go_path[0]):
            continue
        min_thres_dis=r[1]/(3*n)
        destination=cord[box]
        if(box==go_path[len(go_path)-1]):
            r,c=return_cord(box)
            flag=False
            # find centre of white box and making it as a centriod
            mask_w=cv2.inRange(arena,LRW,URW)
            contours, _ = cv2.findContours(mask_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area_of_cnt = cv2.contourArea(cnt)
                if (area_of_cnt > 20):
                    cen = cv2.moments(cnt)
                    cenx = int(cen["m10"] / cen["m00"])
                    ceny = int(cen["m01"] / cen["m00"])
                    r1 = cenx // row
                    c1 = ceny // col
                    if(r1==r & c1==c):
                        flag=True
                        break
            if flag==True:
                flag2=False
                destination=np.array([ceny,cenx])
                min_thres_dis=20# caliberated distance
            else:
                min_thres_dis=20# caliberated distance
                flag2=True

        while (dis > min_thres_dis):  # threshold distance by calibertaion
            p, b_c = aruco(frame)
            bv = bot_vector(p,i,j)
            dv = dirn_of_mov_vector(b_c, destination, r)
            angle = cross_pro(bv, dv)
            while (angle > 10 | angle < -10):
                if (angle <=-10):
                    #   small right turn
                elif angle>10:
                    # small left turn
            # arduino code to move forward little bit
            p, b_c = aruco(frame)
            dis=find_dis(b_c, destination)
            frame=dyn_adj.return_frame()

        if(flag2):
            # arduino code to move hoop up
        else:
            #arduino to pick up the bx

# updating color of horcurex and jail
def update_color_shape(cell_id,lr,ur,num):
    ret, frame = cap.read()
    arena = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    r,c=return_cord(cell_id)
    mask = cv2.inRange(arena, lr, ur)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for ignoring small contours
    total_cnt_area = 0
    no_of_cnt = 0
    for cnt in contours:
        total_cnt_area += cv2.contourArea(cnt)
        no_of_cnt += 1

    avg_cnt_area = total_cnt_area / no_of_cnt
    thresh_cnt_area = avg_cnt_area / 3
    for cnt in contours:
        area_of_cnt = cv2.contourArea(cnt)
        if (area_of_cnt > thresh_cnt_area):
            cen = cv2.moments(cnt)
            cenx = int(cen["m10"] / cen["m00"])
            ceny = int(cen["m01"] / cen["m00"])
            cenx = cenx // n
            ceny = ceny // n
            if (cell_id==cell_num[ceny][cenx]):
                color_mat[ceny][cenx] = num
            # shape recog
                rect = cv2.minAreaRect(cnt)  # making min_area_rect aroung contours
                area_of_rect = rect[1][0] * rect[1][1]  # area of contours
                box = cv2.boxPoints(rect)  # recovering 4 point of min_rect
                box = np.int0(box)
                cv2.drawContours(mask, [box], 0, (100, 100, 255), 2)  # drawing rectangle around contours
                # cv2.imshow('area_of',mask)
                # cv2.waitKey(100)
                rat = area_of_cnt / area_of_rect  # taking ratio of (area of conotur/area of rectangle)
                if rat >= 0.87:
                    print(rat, 1)
                    shape_mat[ceny][cenx] = 1
                else:
                    print(rat, 0)
                    shape_mat[ceny][cenx] = 0
                return True
    return False

