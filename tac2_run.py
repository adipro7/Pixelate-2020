import numpy as np
import cv2
import cv2.aruco as aruco
import serial
import heapq
import all_functions as fuck
import math
import time
from time import sleep

# loading caliberataed things
lr = np.load('lrr.npy')
ur = np.load('urr.npy')
ly = np.load('lry.npy')
uy = np.load('ury.npy')
print(ly, uy)
lb = np.load('lrb.npy')
ub = np.load('urb.npy')
lg = np.load('lrg.npy')
ug = np.load('urb.npy')
lw = np.load('lrw.npy')
uw = np.load('urw.npy')
shape_mat = np.load('shape_mat.npy')
color_mat = np.load('color_mat.npy')
roi = np.load('roi.npy')
print("everything loaded")
print(color_mat)
print(shape_mat)
print(roi)
# starting serial
ser = serial.Serial('COM5', 9600)
sleep(2)
print("connected")
cap = cv2.VideoCapture(1)
# variable declaration
cols = 1
n = 5
rows = n * n + 1
total_cells = n * n + 1

var_cell_wei = [[0 for i in range(cols)] for j in range(rows)]
cell_adj = [[0 for i in range(cols)] for j in range(rows)]
cell_wei = [[0 for i in range(cols)] for j in range(rows)]
cell_center = []  # for storing center of cell
cell_center.append([0, 0])
# numpy array declaration
cell_cord = []
cell_cord.append((0, 0))  # as 0 is not numbering of any cell
cell_num = np.zeros((n, n), dtype=np.int16)
weight_mat = np.zeros((n, n), dtype=np.int16)  # for no of corners
cnt = 1
# numbering of cells of arena box
for i in range(n):
    for j in range(n):
        cell_num[i][j] = cnt  # cell numbering
        cnt += 1
        cell_cord.append((i, j))  # storing value of rows and columns in cell_id
        cell_center.append([(roi[2] // n) * (j + (1 / 2)), (roi[3] // n) * (
                    i + (1 / 2))])  # finding centres of each cells of arena row=r[1] col=col[1]

print(cell_num)
print(cell_cord)
print(cell_center)

# finding cordinate of any cell
def return_cord(cell_id, n=5):
    for i in range(n):
        for j in range(n):
            if (cell_num[i][j] == cell_id):
                return (i, j)

# updating color of horcurex and jail
def update_color_shape(cell_id, lwr, upr, num):
    ret, frame = cap.read()
    arena = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    r, c = return_cord(cell_id)
    mask = cv2.inRange(arena, lwr, upr)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.destroyWindow('mask')
    # for ignoring small contours
    total_cnt_area = 0
    no_of_cnt = 0
    for cnt in contours:
        total_cnt_area += cv2.contourArea(cnt)
        no_of_cnt += 1

    avg_cnt_area = total_cnt_area / no_of_cnt
    thresh_cnt_area = avg_cnt_area / 8
    for cnt in contours:
        area_of_cnt = cv2.contourArea(cnt)
        if (area_of_cnt > thresh_cnt_area):
            cen = cv2.moments(cnt)
            cenx = int(cen["m10"] / cen["m00"])
            ceny = int(cen["m01"] / cen["m00"])
            cenx = cenx // col
            ceny = ceny // row
            if (cell_id == cell_num[ceny][cenx]):
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
                if rat >= 0.85:
                    print(rat, 1)
                    shape_mat[ceny][cenx] = 1
                else:
                    print(rat, 0)
                    shape_mat[ceny][cenx] = 0
                return True
    return False

def find_boat_centre():
    while (True):
        print("here4")
        ret, img = cap.read()
        if (ret == False):
            break
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        arena = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        # zcv2.imshow('arena',arena)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(arena, aruco_dict, parameters=parameters)
        if ids is None:
            continue
        img = cv2.circle(arena, (corners[0][0][0][0], corners[0][0][0][1]), 5, (0, 0, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(10)
        cv2.destroyWindow('img')
        # print("corners")
        # print(corners)
        for x in range(0, ids.shape[0]):
            p1 = max(corners[x][0][0][0], corners[x][0][1][0],
                     corners[x][0][2][0], corners[x][0][3][0])
            p2 = min(corners[x][0][0][0], corners[x][0][1][0],
                     corners[x][0][2][0], corners[x][0][3][0])
            q1 = max(corners[x][0][0][1], corners[x][0][1][1],
                     corners[x][0][2][1], corners[x][0][3][1])
            q2 = min(corners[x][0][0][1], corners[x][0][1][1],
                     corners[x][0][2][1], corners[x][0][3][1])
            xc = int(p2 + abs(p1 - p2) / 2)
            yc = int(q2 + abs(q1 - q2) / 2)
            return corners, [xc, yc]

# for update of weights of which is decided by color and shape of weapons
def update_weight(var_cell_wei, shape, color, n=5):
    u_range = n * n + 1  # 82 in case of n=9
    for i in range(1, u_range, 1):
        for num in cell_adj[i]:
            if (num == 0):
                continue
            else:
                r, c = return_cord(num)
                if (color_mat[r][c] == color and shape_mat[r][c] == shape):
                    var_cell_wei.append(0)  # putting 0 for same color and shape
                else:
                    var_cell_wei.append(1)  # putting 1 for different shape or color


def path(source, destination, wei=cell_wei, n=5):
    path = []
    par = []
    dis = []
    vis = []
    nodes = n * n + 1
    for i in range(nodes):
        dis.append(1000000)
        vis.append(False)
        par.append(-1)
    dis[source] = 0
    par[source] = 0
    q = []
    heapq.heappush(q, (0, source))
    while q:
        next_item = heapq.heappop(q)
        node = next_item[1]
        # print(node)
        if vis[node]:
            continue
        vis[node] = True
        i = 1
        flag = False
        for item in cell_adj[node]:
            if item != 0:
                if (dis[item] > (dis[node] + cell_wei[node][i])):
                    dis[item] = dis[node] + cell_wei[node][i]
                    par[item] = node
                    heapq.heappush(q, (dis[item], item))
                i = i + 1
    # print("parent")
    # print(destination)
    if (par[destination] == -1):
        return path
    path.append(destination)
    while (par[destination] != 0):
        # print(par[destination])
        path.append(par[destination])
        destination = par[destination]
    path.reverse()
    # print(path)
    return path

# to connect a cell to its four neighbour whichever exists
def connect_edges(cell_id):
    r, c = return_cord(cell_id)
    upx = r - 1
    dwx = r + 1
    lefy = c - 1
    ry = c + 1
    if (upx >= 0):
        cell_adj[cell_id].append(cell_num[upx][c])
        cell_wei[cell_id].append(1)
    if (dwx < n):
        cell_adj[cell_id].append(cell_num[dwx][c])
        cell_wei[cell_id].append(1)
    if (lefy >= 0):
        cell_adj[cell_id].append(cell_num[r][lefy])
        cell_wei[cell_id].append(1)
    if (ry < n):
        cell_adj[cell_id].append(cell_num[r][ry])
        cell_wei[cell_id].append(1)

# for remove edges of horcurex when weapons reaches to Horcurex

def check(prison):
    r, c = return_cord(prison)
    upx = r - 1
    dwx = r + 1
    lefy = c - 1
    ry = c + 1
    j = 0
    if (upx >= 0):
        if (color_mat[upx][c] < 2 and color_mat[upx][c] >= 0):
            return True
    if (dwx < n):
        if (color_mat[dwx][c] < 2 and color_mat[dwx][c] >= 0):
            return True
    if (lefy >= 0):
        if (color_mat[r][lefy] < 2 and color_mat[r][lefy] >= 0):
            return True
    if (ry < n):
        if (color_mat[r][ry] < 2 and color_mat[dwx][c] >= 0):
            return True
    return False

def remove_edges(cell_id):
    if cell_adj[cell_id] is None:
        return
    while (len(cell_adj[cell_id]) > 1):
        cell_adj[cell_id].pop()
        cell_wei[cell_id].pop()

# return front direction of boat
def bot_vector(p, i, j):
    return (p[0][0][i][0] - p[0][0][j][0], p[0][0][i][1] - p[0][0][j][1])


# return direction in which boat have to move
def dirn_of_mov_vector(boat_centre, next_cell):
    # cell_cent=cell_center[next_cell]
    return (next_cell[0] - boat_centre[0], next_cell[1] - boat_centre[1])

# return distance between boat_centre and cell_cent
def find_dis(boat_centre, next_cell):
    # cell_cent=cell_center[next_cell]
    print("next_cell :", next_cell, boat_centre)
    return math.sqrt(((next_cell[0] - boat_centre[0]) ** 2) + ((next_cell[1] - boat_centre[1]) ** 2))

# determine angle between boat direction and direction of movemnet
def cross_pro(dirn_of_mov=[1, 1], boat_vector=[0, 1]):
    a = np.array(dirn_of_mov)
    b = np.array(boat_vector)
    # print(np.cross(a, b))
    mag = (math.sqrt(dirn_of_mov[0] ** 2 + dirn_of_mov[1] ** 2)) * (
        math.sqrt(boat_vector[0] ** 2 + boat_vector[1] ** 2))
    # print(math.degrees(math.asin(np.cross(a,b)/mag)))
    return (math.degrees(math.asin(np.cross(a, b) / mag)))

# determining measure of angle to turn
def dot_pro(dirn_of_mov, boat_vector):
    a = np.array(dirn_of_mov)
    b = np.array(boat_vector)
    # print(np.cross(a, b))
    mag = (math.sqrt(dirn_of_mov[0] ** 2 + dirn_of_mov[1] ** 2)) * (math.sqrt(boat_vector[0] ** 2 + boat_vector[1] ** 2))
    # print(math.degrees(math.asin(np.cross(a,b)/mag)))
    return (math.degrees(math.acos(np.dot(a, b) / mag)))

# locomaion of boat
def bot_movement(go_path):
    flag2 = False  # if true then to move hoop that is next cell is blue
    flag = False  # if true then down hoop to that is next cell is green
    for box in go_path:
        print("visiting :", box)
        dis = 10000
        # ret,img=cap.read()
        # arena = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        if (box == go_path[0]):
            continue
        min_thres_dis = roi[2] / (2 * n)  # distance is less than length_of_one_side/3
        # print("min_threshold_dis :",min_thres_dis)
        destination = cell_center[box]
        print("box :", box)
        # print("destinstion :",destination)
        if (box == go_path[len(go_path) - 1]):
            #break
            r, c = return_cord(box)
            r1 = -1
            c1 = -1
            min_thres_dis=(roi[2]//n)*25//25# if overshoot increase 10

            # find centre of white box and making it as a centriod
            ret, img = cap.read()
            arena = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            mask_w = cv2.inRange(arena, lw, uw)
            contours, _ = cv2.findContours(mask_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # for ignoring small contours
            total_cnt_area = 0
            no_of_cnt = 0
            for cnt in contours:
                total_cnt_area += cv2.contourArea(cnt)
                no_of_cnt += 1
            avg_cnt_area = total_cnt_area / no_of_cnt
            thresh_cnt_area = avg_cnt_area / 5

            for cnt in contours:
                area_of_cnt = cv2.contourArea(cnt)
                if (area_of_cnt > thresh_cnt_area):
                    cen = cv2.moments(cnt)
                    cenx = int(cen["m10"] / cen["m00"])
                    ceny = int(cen["m01"] / cen["m00"])
                    c1 = cenx // n
                    r1 = ceny // n
                    if (r1 == r and c1 == c):
                        flag = True
                        break
            if flag == True:
                flag2 = False
                destination = np.array([r1, c1])
                #min_thres_dis = (roi[2] / (1.5 * n))  # caliberated distance
                min_thres_dis =  min_thres_dis=(roi[2]//n)*20//25
            else:
                flag2 = True
                min_thres_dis =  min_thres_dis=(roi[2]//n)*20//25

            if (color_mat[r][c] == 2):
                flag2 = False
                flag = True
                #min_thres_dis = (roi[2] / (2 * n))  # caliberated distance
        while (dis > min_thres_dis):  # threshold distance by calibertaion
            # ret,frame=cap.read()
            print("dis :", dis)
            p, b_c = find_boat_centre()
            bv = bot_vector(p, 0, 3)
            dv = dirn_of_mov_vector(b_c, destination)
            angle_sign = cross_pro(dv,bv)
            angle_mag = dot_pro(dv, bv)
            # print("ang :", angle)
            while (int(angle_mag) > 10 or int(angle_mag) < -10):
                print("ang :", angle_mag)
                if (int(angle_sign) <= 0):  # small right turn
                    ser.write(b'r')
                    sleep(.3)
                    ser.write(b's')
                    sleep(0.4)
                elif (int(angle_sign) >0):
                    ser.write(b'l')
                    sleep(.3)
                    ser.write(b's')
                    sleep(0.4)
                p, b_c = find_boat_centre()
                bv = bot_vector(p, 0, 3)
                dv = dirn_of_mov_vector(b_c, destination)
                angle_sign = cross_pro(dv, bv)
                angle_mag = dot_pro(dv, bv)
            # arduino code to move forward little bit
            ser.write(b'f')
            sleep(.4)
            ser.write(b's')
            sleep(.4)
            # ret,frame=cap.read()
            p, b_c = find_boat_centre()
            # print("bc :",b_c)
            dis = find_dis(b_c, destination)

    print(flag)
    print(flag2)
    if(flag2):
        ser.write(b'D')
        sleep(0.4)
        #ser.write(b'D')
        #sleep(0.4)
        # arduino code to move hoop up

    else:
        ser.write(b'U')
        sleep(0.4)
        #ser.write(b'U')
        #sleep(0.4)
        #arduino to pick up the bx

# finding horcurex and jail
horcruxes = []
azkaban_prison = []
weapons = []
green_cell = []
free_jail = []  # 3 jail together
closed_jail = []  # 1 jail closed

for i in range(n):
    for j in range(n):
        print(color_mat[i][j])
        if (color_mat[i][j] == -4):  # white present on horcruex
            horcruxes.append(cell_num[i][j])
        if (color_mat[i][j] == 2):  # jail
            if (check(cell_num[i][j])):
                free_jail.append(cell_num[i][j])
            else:
                closed_jail.append(cell_num[i][j])
        if (color_mat[i][j] == 5):  # weapons
            weapons.append(cell_num[i][j])
        if (color_mat[i][j] == 4 or color_mat[i][j] == -4):
            green_cell.append(cell_num[i][j])

print("horcurex :", horcruxes)
print("free_jail :", free_jail)
print('closed_jail :', closed_jail)
print("green_cell", green_cell)
print("weapons :", weapons)

# making cell_wei and cell_adj
for i in range(n):
    for j in range(n):
        upx = i - 1
        dwx = i + 1
        lefy = j - 1
        ry = j + 1
        if (color_mat[i][j] <= 2 and color_mat[i][j] >= 0):
            if (upx >= 0):
                cell_adj[cell_num[i][j]].append(cell_num[upx][j])
                cell_wei[cell_num[i][j]].append(1)
            if (dwx < n):
                cell_adj[cell_num[i][j]].append(cell_num[dwx][j])
                cell_wei[cell_num[i][j]].append(1)
            if (lefy >= 0):
                cell_adj[cell_num[i][j]].append(cell_num[i][lefy])
                cell_wei[cell_num[i][j]].append(1)
            if (ry < n):
                cell_adj[cell_num[i][j]].append(cell_num[i][ry])
                cell_wei[cell_num[i][j]].append(1)
print(cell_adj)

# pass source position and destination position :taking 3 boxes to jail
horcruex_counter = 0
jail_counter = 0
weapons_counter = 0
left_jail = []
left_horcruex = []
left_weapons = []
vis_horcruex = []  # bool var to maintain visited of horcruex
for i in range(n * n + 1):
    vis_horcruex.append(False)

while (True):
    row = math.ceil(roi[2] / n)
    col = math.ceil(roi[2] / n)
    print(row, col)
    print(roi[2])
    for boxes in (horcruxes):
        p, bc = find_boat_centre()
        print(bc)
        cellid = cell_num[int(bc[1] // col)][int(bc[0] // row)]
        print(cellid)
        # print()
        print(cell_num[(bc[1] // col)][(bc[0] // row)], boxes)
        path_to_horcruex = path(cell_num[(bc[1] // col)][(bc[0] // row)], boxes)
        print("path_to_horcurex :", path_to_horcruex)
        bot_movement(path_to_horcruex)
        ser.write(b's')
        connect_edges(boxes)
        time.sleep(1)

        p, bc = find_boat_centre()
        cellid = cell_num[int(bc[1] // col)][int(bc[0] // row)]
        print(cellid)
        path_to_jail = path(cell_num[(bc[1] // col)][(bc[0] // row)],
                            free_jail[jail_counter])  # path from horcruex to jail
        print("path_to_jail :", path_to_jail)
        bot_movement(path_to_jail)
        ser.write(b's')
        sleep(0.4)
        ser.write(b'b')
        sleep(1.5)
        ser.write(b's')
        sleep(0.2)

        if (update_color_shape(boxes, lr, ur, 0)):
            print("hel")
        else:
            temp = update_color_shape(boxes, ly, uy, 1)
        # remove_edges(free_jail[jail_counter])
        jail_counter += 1
        horcruex_counter += 1
        # move bot two step back as jail is disconnected
        # ser.write(b'b')
        if (jail_counter == len(free_jail)):
            break
        if (horcruex_counter == len(horcruxes)):
            break

    for ele in (weapons):
        p, bc = find_boat_centre()
        path_to_weapons = path(cell_num[(bc[1] // col)][(bc[0] // row)], ele)
        print("path_to_weapons :", path_to_weapons)
        bot_movement(path_to_weapons)
        ser.write(b's')
        time.sleep(0.2)
        ser.write(b'b')
        sleep(2.5)
        ser.write(b's')
        sleep(0.2)

        '''if (last_horcruex != 0):
            fuck.remove_edges(cell_wei, cell_adj, last_horcruex)'''

        # arduino code to move two box backward
        # ser.write(b'B')
        # code to scan frame
        print("till sucess")
        print(ele, lr, ur)
        print(ele, ly, uy)
        if (update_color_shape(ele, lr, ur, 0)):
            print("hel")

        elif (update_color_shape(ele, ly, uy, 1)):
            r, c = return_cord(ele)
            print(color_mat[r][c], shape_mat[r][c])
        _f = update_color_shape(1, ly, uy, 0)
        _e = update_color_shape(5, ly, uy, 0)
        _d = update_color_shape(21, ly, uy, 0)
        _c = update_color_shape(1, ly, uy, 1)
        _a = update_color_shape(5, ly, uy, 1)
        _b = update_color_shape(21, ly, uy, 1)
        ser.write(b'f')
        sleep(2.5)
        ser.write(b's')
        sleep(0.4)

        print("here yellow")
        print(color_mat)
        print(shape_mat)
        # temp = update_color_shape(ele,ly,uy,1)
        # first of all move box to current weapons place
        # ser.write(b'F')
        # flag for making sure if it find its color and shape with horcruex
        print("hel")
        flag = False  # initially we believe that there is no match
        for boxes in (green_cell):
            r, c = return_cord(boxes)
            r1, c1 = return_cord(ele)
            print("debug")
            print(r, c)
            print(r1, c1)
            if (color_mat[r][c] == color_mat[r1][c1] and shape_mat[r][c] == shape_mat[r1][c1]):
                print("matched")
                flag = True
                vis_horcruex[boxes] = True
                connect_edges(ele)
                # update weight of graph
                var_cell_wei = [[0 for i in range(cols)] for j in range(rows)]
                update_weight(var_cell_wei, shape_mat[r1][c1], color_mat[r1][c1], n)
                p, bc = find_boat_centre()
                path_to_horcruex = path(ele, boxes, wei=var_cell_wei, n=5)  # if error replace find_boat_center by ele
                print(path_to_horcruex)
                bot_movement(path_to_horcruex)
                ser.write(b's')
                sleep(1)
                # remove_edges(boxes)
                color_mat[r][c] = 10  # to sure that it does not match with any other else
                # move bot two step back or either bot centre is not in that horcruex
                # ser.write(b'B')
    break

cap.release()
cv2.destroyAllWindows()