import numpy as np
import cv2
import cv2.aruco as aruco
import serial
import heapq
import all_functions as fuck
import math
from time import sleep


lr=np.load('lrr.npy')
ur=np.load('urr.npy')
ly=np.load('lry.npy')
uy=np.load('ury.npy')
lb=np.load('lrb.npy')
ub=np.load('urb.npy')
lg=np.load('lrg.npy')
ug=np.load('urb.npy')
lw=np.load('lrw.npy')
uw=np.load('urw.npy')

#color_mat=np.load('')
#shape_mat=np.load('')
roi=np.load('roi.npy')
print("everything loaded")

serial.begin('com5',9600)
sleep(2)
print("connected")
ser=serial.Serial()

# numpy array declaration
cols=1
n=9
rows=n*n+1


var_cell_wei = [[0 for i in range(cols)] for j in range(rows)]
cell_adj = [[0 for i in range(cols)] for j in range(rows)]
cell_wei = [[0 for i in range(cols)] for j in range(rows)]

cord=[]
# numpy array declaration
cell_num=np.zeros((n, n), dtype=np.int16)
shape_mat = np.full((n,n),-1)  # for shape
weight_mat = np.zeros((n, n), dtype=np.int16)  # for no of corners
color_mat = np.zeros((n, n), dtype=np.int16)  # for color
cnt=1
# numbering of cells of arena box
for i in range(n):
    for j in range(n):
        cell_num[i][j]=cnt
        cnt+=1

cap=cv2.VideoCapture(1)
print(cell_num)
# finding cordinate of any cell
def return_cord(cell_id,n=9):
    for i in range(n):
        for j in range(n):
            if(cell_num[i][j]==cell_id):
                return (i,j)


# updating color of horcurex and jail
def update_color_shape(cell_id, lr, ur, num):
    ret, frame = cap.read()
    arena = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    r, c = return_cord(cell_id)
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
                if rat >= 0.87:
                    print(rat, 1)
                    shape_mat[ceny][cenx] = 1
                else:
                    print(rat, 0)
                    shape_mat[ceny][cenx] = 0
                return True
    return False

# for making color and shape matrix
def color_shape(lrg,urg,arena,num,boat_center):
    mask = cv2.inRange(arena, lrg, urg)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for ignoring small contours
    total_cnt_area=0
    no_of_cnt=0
    for cnt in contours:
        total_cnt_area +=cv2.contourArea(cnt)
        no_of_cnt+=1

    avg_cnt_area=total_cnt_area/no_of_cnt
    thresh_cnt_area=avg_cnt_area/3
    for cnt in contours:
        area_of_cnt = cv2.contourArea(cnt)
        if (area_of_cnt>thresh_cnt_area):
            cen = cv2.moments(cnt)
            cenx = int(cen["m10"] / cen["m00"])
            ceny = int(cen["m01"] / cen["m00"])
            cenx = cenx // n
            ceny = ceny // n
            if boat_center!=(ceny,cenx):
                if(num==4):# to become sure that there is death eater above green cell
                    if(color_mat[ceny][cenx]==5):
                        color_mat[ceny][cenx]=4
                    green_cell.append(cell_num[ceny][cenx])
                else:
                    color_mat[ceny][cenx] = num
            else:
                color_mat[ceny][cenx] = 2
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

def find_boat_centre():
    while(1):
        ret, frame = cap.read()
        arena = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(arena, aruco_dict, parameters=parameters)
        if ids is None:
            continue
        img=cv2.circle(arena,(corners[0][0][0][0],corners[0][0][0][1]),5,(0,0,255),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
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
            return corners,[xc,yc]

# for update of weights of which is decided by color and shape of weapons
def update_weight(var_cell_wei,shape,color,n=9):
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

def path(source,destination,wei=cell_wei,n=9):
    path=[]
    par=[]
    dis=[]
    vis=[]
    nodes=n*n+1
    for i in range(nodes):
        dis.append(1000000)
        vis.append(False)
        par.append(-1)
    dis[source]=0
    par[source]=0
    q=[]
    heapq.heappush(q,(0,source))
    while q:
        next_item = heapq.heappop(q)
        node=next_item[1]
        #print(node)
        if vis[node]:
            continue
        vis[node]=True
        i=1
        flag=False
        for item in cell_adj[node]:
           if item!=0:
               if(dis[item]>(dis[node]+cell_wei[node][i])):
                    dis[item]=dis[node]+cell_wei[node][i]
                    par[item] = node
                    heapq.heappush(q,(dis[item],item))
               i=i+1
    #print("parent")
    #print(destination)
    if(par[destination]==-1):
        return path
    path.append(destination)
    while(par[destination]!=0):
        #print(par[destination])
        path.append(par[destination])
        destination=par[destination]
    path.reverse()
    return path

# finding centres of each cells of arena
def find_centre(n=9):
    ret, frame = cap.read()
    arena = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    l,b=arena.shape
    row=l/n
    col=b/n
    for i in range(n):
        for j in range(n):
            cord.append([col*(i+(1/2)),row*(j+(1/2))])

# return front direction of boat
def bot_vector(p,i,j):
    return p[0][0][i]-p[0][0][j]

# return direction in which boat have to move
def dirn_of_mov_vector(boat_centre,next_cell,roi):
    cell_cent=find_centre(next_cell)
    boat_centre[0]-=roi[0]
    boat_centre[1]-=roi[1]# source of error y and x
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
    return (math.degrees(math.asin(np.cross(a,b)/mag)))

# locomaion of boat
def bot_movement(go_path):
    dis = 10000
    flag2=False# if true then to move hoop that is next cell is blue
    flag = False # if true then down hoop to that is next cell is green
    for box in (go_path):
        if(box==go_path[0]):
            continue
        min_thres_dis=roi[2]/(3*n)# distance is less than length_of_one_side/3
        destination=cord[box]
        if(box==go_path[len(go_path)-1]):
            r,c=return_cord(box)
            r1=0
            c1=0
            # find centre of white box and making it as a centriod
            mask_w=cv2.inRange(arena,lw,uw)
            contours, _ = cv2.findContours(mask_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area_of_cnt = cv2.contourArea(cnt)
                if (area_of_cnt > 20):
                    cen = cv2.moments(cnt)
                    cenx = int(cen["m10"] / cen["m00"])
                    ceny = int(cen["m01"] / cen["m00"])
                    c1 = cenx // n
                    r1 = ceny // n
                    if(r1==r & c1==c):
                        flag=True
                        break
            if flag==True:
                flag2=False
                destination=np.array([r1,c1])
                min_thres_dis=(roi[2]/(2*n))# caliberated distance
            else:
                min_thres_dis=20# caliberated distance
                flag2=True

        while (dis > min_thres_dis): # threshold distance by calibertaion
            #ret,frame=cap.read()
            p, b_c = find_boat_centre()
            bv = bot_vector(p,i,j)
            dv = dirn_of_mov_vector(b_c,destination,roi)
            angle = cross_pro(bv, dv)
            while (angle > 10 | angle < -10):
                if (angle <=-10):#small right turn
                    ser.write(b'r')
                    sleep(0.04)
                    ser.write(b's')
                elif (angle>10):
                    ser.write(b'l')
                    sleep(0.04)
                    ser.write(b's')
                p, b_c = find_boat_centre()
                bv = bot_vector(p, i, j)
                dv = dirn_of_mov_vector(b_c, destination, roi)
                angle = cross_pro(bv, dv)
            # arduino code to move forward little bit
            ser.write(b'f')
            sleep(0.1)
            ser.write(b's')
            #ret,frame=cap.read()
            p, b_c = find_boat_centre()
            dis=find_dis(b_c, destination,roi)

        if(flag2):
            # arduino code to move hoop up

        else:
            #arduino to pick up the bx

# finding horcurex and jail
horcruxes = []
azkaban_prison = []
weapons = []
green_cell=[]
# making matrix of color and shape

boat_center = find_boat_centre()
ret,frame=cap.read()
arena = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
color_shape(lr, ur, arena, 0,boat_center)
color_shape(ly, uy, arena, 1,boat_center)
color_shape(lw, uw, arena, 5,boat_center)
color_shape(lb, ub, arena, 2,boat_center)
color_shape(lg, ug, arena, 4,boat_center)

# to store the coordinate of center of each cell
find_centre(n)


for i in range(n):
    for j in range(n):
        if(color_mat[i][j]==4):
            horcruxes.append(cell_num[i][j])
        if(color_mat[i][j]==2):
            azkaban_prison.append(cell_num[i][j])
        if (color_mat[i][j] == 5):
            weapons.append(cell_num[i][j])

# making cell_wei and cell_adj
for i in range(n):
    for j in range(n):
        upx=i-1
        dwx=i+1
        ly=j-1
        ry=j+1
        if(color_mat[i][j]<=2):
            if(upx>=0):
                cell_adj[cell_num[i][j]].append(cell_num[upx][j])
                cell_wei[cell_num[i][j]].append(1)
            if(dwx<n):
                cell_adj[cell_num[i][j]].append(cell_num[dwx][j])
                cell_wei[cell_num[i][j]].append(1)
            if (ly >=0):
                cell_adj[cell_num[i][j]].append(cell_num[i][ly])
                cell_wei[cell_num[i][j]].append(1)
            if (ry < n):
                cell_adj[cell_num[i][j]].append(cell_num[i][ry])
                cell_wei[cell_num[i][j]].append(1)



# pass source position and destination position :taking 3 boxes to jail
horcruex_counter=0
jail_counter=0
weapons_counter=0
left_jail=[]
left_horcruex=[]
left_weapons=[]
vis_horcruex=[]# bool var to maintain visited of horcruex
for i in range(n*n+1):
    vis_horcruex.append(False)

for boxes in (horcruxes):
    path_to_horcruex=path(find_boat_centre(),boxes)
    if(len(path_to_horcruex)!=0):
        horcruex_counter+=1

    flag=True# denoting there is any path from horcurex to jail
    path_to_jail=[]# path from horcruex to jail
    while(flag & jail_counter<len(azkaban_prison)):
        path_to_jail=path(find_boat_centre(),azkaban_prison[jail_counter])
        if(len(find_boat_centre())!=0):# means there is path to jail
            jail_counter += 1
            flag=False
            break
        else:
            left_jail.append(jail_counter)
            jail_counter+=1
    if(flag):# no path is left from horcruex to jail
        horcruex_counter-=1
        break
    bot_movement(path_to_horcruex)
    fuck.connect_edges(cell_adj, cell_wei, cell_num, boxes, n=9)
    bot_movement(path_to_jail)
    if(update_color_shape(boxes,lr,ur,0)):
        print("hel")
    else:
        update_color_shape(boxes, ly, uy, 1)
    fuck.remove_edges(cell_wei,cell_adj,azkaban_prison[jail_counter-1])
    # move bot two step back as jail is disconnected
    if(jail_counter==len(azkaban_prison)):
        break
    if(horcruex_counter==len(horcruxes)):
        break

last_horcruex=0
for ele in (weapons):
    path_to_weapons=path(find_boat_centre(),ele)
    bot_movement(path_to_weapons)
    '''if (last_horcruex != 0):
        fuck.remove_edges(cell_wei, cell_adj, last_horcruex)'''

    # arduino code to move two box backward

    # code to scan frame
    if(update_color_shape(ele,lr,ur,0)):
        print("hel")
    else:
        update_color_shape(ele, ly, uy, 1)
    # first of all move box to current weapons place

    # flag for making sure if it find its color and shape with horcruex
    flag=False# initially we believe that there is no match
    for boxes in (green_cell):
        r,c=return_cord(boxes)
        r1,c1=return_cord(ele)
        if(color_mat[r][c]==color_mat[r1][c1] & shape_mat[r][c]==shape_mat[r1][c1] & vis_horcruex[boxes]==False):
            flag=True
            vis_horcruex[boxes]=True
            fuck.connect_edges(cell_adj,cell_wei,cell_num,ele,n=9)
            #update weight of graph
            var_cell_wei=[[0 for i in range(cols)] for j in range(rows)]
            update_weight(var_cell_wei, shape_mat[r1][c1],color_mat[r1][c1],n)
            path_to_horcruex=path(find_boat_centre(),boxes,wei=var_cell_wei,n=9)# if error replace find_boat_center by ele
            bot_movement(path_to_horcruex)
            fuck.remove_edges(cell_wei, cell_adj, last_horcruex)
            #move bot two step back or either bot centre is not in that horcruex
            last_horcruex=boxes
    if(flag==False):
        left_weapons.append(ele)

#fuck.remove_edges(cell_wei, cell_adj, last_horcruex)# to disconnect horcruex from its neighbour if weapons is placed

# move bot to last left horcruex
for i in range(horcruex_counter,len(horcruxes),1):
    path_to_horcruex=path(find_boat_centre(),horcruxes[horcruex_counter])
    bot_movement(path_to_horcruex)
    fuck.connect_edges(cell_adj,cell_wei,cell_num,cell_id,n=9)
    for left in (left_jail):
        path_to_jail=path(find_boat_centre(),left)
        bot_movement(path_to_jail)
        fuck.remove_edges(cell_wei, cell_adj, cell_id=jail)
        # remove bot some step back as jai is disconnected

for ele in (left_weapons):
    path_to_weapons=path(find_boat_centre(),ele)
    bot_movement(path_to_weapons)
    if (last_horcruex != 0):
    fuck.remove_edges(cell_wei, cell_adj, last_horcruex)

    # arduino code to move two box backward

    # code to scan frame
    if(update_color_shape(ele,lr,ur,0)):
        print("hel")
    else:
        update_color_shape(ele, ly, uy, 1)

    # first of all move box to current weapons place
    flag = False  # initially we believe that there is no match
    for boxes in (green_cell):
        r, c = return_cord(boxes)
        r1, c1 = return_cord(ele)
        if (color_mat[r][c] == color_mat[r1][c1] & shape_mat[r][c] == shape_mat[r1][c1] & vis_horcruex[boxes]=False):
            flag = True
            vis_horcruex[boxes] = True
            fuck.connect_edges(cell_adj, cell_wei, cell_num, ele, n=9)
            # update weight of graph
            var_cell_wei = [[0 for i in range(cols)] for j in range(rows)]
            update_weight(var_cell_wei, shape_mat[r1][c1], color_mat[r1][c1], n)
            path_to_horcruex = path(find_boat_centre(), boxes, wei=var_cell_wei,
                                    n=9)  # if error replace find_boat_center by ele
            bot_movement(path_to_horcruex)
            last_horcruex = boxes



