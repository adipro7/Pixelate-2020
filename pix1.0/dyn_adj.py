import numpy as np
import cv2
import heapq
import adjmat_wei

def return_frame():
    ret,frame=cap.read()
    return frame
def check(prison):
    r,c = return_cord(prison)
    upx = r - 1
    dwx = r + 1
    ly = c - 1
    ry = c + 1
    j = 0
    if(upx>=0):
        if(color_mat[upx][c]<2):
            return True
    if(dwx<n):
        if(color_mat[dwx][c]<2):
            return True
    if(ly>=0):
        if(color_mat[r][ly]<2):
            return True
    if ry<n:
        if color_mat[r][ry]<2:
            return True
    return False

def return_cord(cell_id):
    for i in range(n):
        for j in range(n):
            if(cell_num[i][j]==cell_id):
                return (i,j)
def path(source,destination,wei=cell_wei):
    path=[]
    par=[]
    dis=[]
    vis=[]
    for i in range(82):
        dis.append(1000000)
        vis.append(False)
        par.append(-1)
    dis[source]=0
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
                    dis[item]=min(dis[item],dis[node]+cell_wei[node][i])
                    par[item] = node
                    heapq.heappush(q,(dis[item],item))
               i=i+1
               if (item == destination):
                   flag=True
                   break
        if (flag):
            break
    #print("parent")
    #print(destination)
    path.append(destination)
    while(par[destination]!=-1):
        #print(par[destination])
        path.append(par[destination])
        destination=par[destination]
    path.reverse()
    return path


color_mat=[[4, 1, 0 ,1, 0, 1, 0, 1, 4],
            [1 ,0 ,1, 0, 0, 0, 1, 0, 1],
            [0 ,0, 1, 1, 1, 1, 0, 0, 0],
            [0 ,1 ,0, 0, 5, 1, 1, 1, 2],
             [1, 0, 1 ,5 ,2, 5, 0, 1, 2],
             [0 ,1 ,1 ,0 ,5 ,0 ,1, 0,2],
             [0 ,0, 1, 1, 1, 0, 1, 1, 0],
             [1 ,1 ,0, 0 ,0 ,1 ,0 ,1 ,0],
             [4, 0, 0 ,1 ,1, 0, 1 ,0 ,4]]
color_mat=np.transpose(color_mat)

rows=82
cols=1
n=9

var_cell_wei = [[0 for i in range(cols)] for j in range(rows)]
cell_adj = [[0 for i in range(cols)] for j in range(rows)]
cell_wei = [[0 for i in range(cols)] for j in range(rows)]

cell_num=np.zeros((9,9),dtype=np.int16)
Horcruxes=[]
Weapons=[]
Azkaban_prison=[]
not_out=[]
horcurex_destroyed=[]# for destroyed horcurex

# cell numbering from 1 to 81
cnt=1
for i in range(9):
    for j in range (9):
        cell_num[i][j]=cnt
        cnt=cnt+1

# denoting Horcruxes,Weapons,and Azkaban_prison
for i in range(n):
    for j in range(n):
        if color_mat[i][j]==2:
            Azkaban_prison.append(cell_num[i][j])
        if color_mat[i][j]==4:
            Horcruxes.append(cell_num[i][j])
            horcurex_destroyed.append(False)
        if color_mat[i][j]==5:
            Weapons.append(cell_num[i][j])
        if color_mat[i][j]>=2:
            not_out.append(cell_num[i][j])

# cell connecting to its adjacent except Azkaban prison with white neighbour ,Weapons and Horcruxes
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

unvis_prison=-1# for unvis_prison btw white box
unvis_horcurex=-1# left horcurex

# kunal algorithm
#print("Horcurex")
#print(Horcruxes)
#print("Azakabian")
#print(Azkaban_prison)

# pass source position and destination position :taking 3 boxes to jail
j=0
for i in range(3):
    if(check(Azkaban_prison[j])==False):
        j=j+1
    go_path=path(Azkaban_prison[j],Horcruxes[i])#source =aruco position
    adjmat_wei.bot_vector(go_path)
    #vector concept and pick up the box
    adjmat_wei.connect_edges(cell_adj,cell_wei,cell_num,Horcruxes[i])# para: adj_list,adj_wei,cell_num,cell_id
    return_path=path(Horcruxes[i],Azkaban_prison[j])#here source will be aruco position
    adjmat_wei.bot_vector(return_path)
    adjmat_wei.remove_edges(cell_wei,cell_adj,Azkaban_prison[j])# remove edges connected to jail
    #print("path")
    print(Azkaban_prison[j],Horcruxes[i])
    print(go_path)
    print(Horcruxes[i],Azkaban_prison[j])
    print(return_path)
    j=j+1
# update color and shape matrix again


# now going to one of the weapons
to_rem_edg_hor=1# check it once
for num in range(Weapons):
    path(source,num)
    # remove weapons 2 boxes back
    # update color_mat and shape_mat due to weapons
    flag=False
    r,c=return_cord(num)
    for ele in (Horcruxes):
        r1,c1=return_cord(ele)
        if color_mat[r1][c1]==color_mat[r][c] & shape_mat[r1][c1]==shape_mat[r][c]:
            to_rem_edg_hor=ele
            # aurdino code to move weapons to its initial position
            adjmat_wei.connect_edges(cell_adj,cell_wei,cell_num,num)
            path_w_h=path(source,ele,var_cell_wei)
            # aurdino code to move weapons from its position to horcurex location
            adjmat_wei.bot_vector(path_w_h)
            flag=True
            break
    if(flag==True):
        break
    if(flag==False):
        # special specified code for aurdino ot move forward keep the weapons and then move to next weapons


# code for 4th horcurex to jail
go_path=path(source,unvis_horcurex)
adjmat_wei.bot_vector(go_path)
adjmat_wei.remove_edges(to_rem_edg_hor)
return_path=path(source,unvis_prison)
adjmat_wei.bot_vector(return_path)

adjmat_wei.connect_edges(cell_adj,cell_wei,cell_num,unvis_horcurex)
adjmat_wei.remove_edges(unvis_prison)

# codee to put all left 3 weapons to horcurex

for num in range(Weapons):
    go_path=path(source,num)
    # aurdino code to move to weapons
    adjmat_wei.bot_vector(return_path)

    adjmat_wei.remove_edges(cell_wei,cell_adj,to_rem_edg_hor)
    # remove weapons 2 boxes back
    # update color_mat and shape_mat due to weapons
    r,c=return_cord(num)
    for ele in (Horcruxes):
        r1,c1=return_cord(ele)
        if color_mat[r1][c1]==color_mat[r][c] & shape_mat[r1][c1]==shape_mat[r][c]:
            to_rem_edg_hor=ele
            # aurdino code to move weapons to its initial position
            adjmat_wei.connect_edges(cell_adj,cell_wei,cell_num,num)
            path_w_h=path(source,ele,var_cell_wei)
            adjmat_wei.bot_vector(path_w_h)
            # aurdino code to move weapons from its position to horcurex location
            break
