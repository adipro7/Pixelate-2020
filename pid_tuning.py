import cv2
import numpy as np
import cv2.aruco as aruco
import math
import serial
from time import sleep
#cap = cv2.VideoCapture("http://192.168.43.63:4747/video")

ser=serial.Serial('COM5',9600)
sleep(2)
print("connected")


def writeToSerial(a, b, c, d):
    ser.write(str.encode(str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + ' 0\n'))
    ser.readline()
    pass
'''
def find_boat_centre():
    while(1):
        ret, arena = cap.read()
        #arena = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
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
            return corners,(xc,yc)
'''
def bot_vector(p,i,j):
    return p[0][0][i]-p[0][0][j]

# return direction in which boat have to move
def dirn_of_mov_vector(boat_centre,next_cell):
    #cell_cent=(next_cell)
    #boat_centre[0]-=roi[0]
    #boat_centre[1]-=roi[1]# source of error y and x
    return next_cell-boat_centre

# determine angle between boat direction and direction of movemnet
def cross_pro(dirn_of_mov=[1,1],boat_vector=[0,1]):
    a=np.array(dirn_of_mov)
    b=np.array(boat_vector)
    print(np.cross(a, b))
    mag = (math.sqrt(dirn_of_mov[0] ** 2 + dirn_of_mov[1] ** 2)) * (math.sqrt(boat_vector[0] ** 2 + boat_vector[1] ** 2))
    print(math.degrees(math.asin(np.cross(a,b)/mag)))
    return (math.degrees(math.asin(np.cross(a,b)/mag)))

KP = 7
KD = 70
KI=0
MaxSpeedLine = 150
BaseSpeedLine = 100
#instantStopAngle = 0.250
#instantStopLine = 0.100

#thresholdForRect = 200

#Initializatins
prev_error = 0
rightMotorSpeed=0
leftMotorSpeed=0
motor_speed=0
#desired_value=(200,100)# set desired value according to camera

set_point=[(96,365),(56,269),(104,168),(274,252)]
i=0
'''
while (True):
    ret, image = cap.read()
    if(ret==False):
        break
    print("here1")
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print("here2")

    p,center_of_bot=find_boat_centre()
    bv=bot_vector(p,0,1)
    dv=dirn_of_mov_vector(center_of_bot,set_point[i])
    error=cross_pro(dv,bv)
    if(abs(error)>80):
        # stop bot
        leftMotorSpeed=0
        rightMotorSpeed=0
        i+=1
        if(i>=4):
            i=i%4

        p, b_c = find_boat_centre()
        bv = bot_vector(p, 0, 1)
        dv = dirn_of_mov_vector(b_c, set_point[i])
        angle = cross_pro(bv, dv)
        while (angle > 10 | angle < -10):
            if (angle <= -10):
            # small right turn
                rightMotorSpeed=40
                leftMotorSpeed=40
                lind=0
                rind=1
                writeToSerial(rightMotorSpeed,leftMotorSpeed,rind,lind)
                sleep(0.05)
            elif angle > 10:
                # small left turn
                rightMotorSpeed = 40
                leftMotorSpeed = 40
                lind = 0
                rind = 1
                writeToSerial(rightMotorSpeed, leftMotorSpeed, rind, lind)
                sleep(0.05)

            p, b_c = find_boat_centre()
            bv = bot_vector(p, 0, 1)
            dv = dirn_of_mov_vector(b_c, set_point[i])
            angle = cross_pro(bv, dv)
        continue
    #integral = integral_prior + error * iteration_time
    #derivative = (error – prev_error) / iteration_time
    motor_speed = KP*error + KD*(error-prev_error)
    rightMotorSpeed = BaseSpeedLine - motor_speed
    leftMotorSpeed = BaseSpeedLine + motor_speed

    if (rightMotorSpeed > MaxSpeedLine): rightMotorSpeed = MaxSpeedLine
    if (leftMotorSpeed > MaxSpeedLine): leftMotorSpeed = MaxSpeedLine
    if (rightMotorSpeed < 0): rightMotorSpeed = 0
    if (leftMotorSpeed < 0): leftMotorSpeed = 0
    print('L:', leftMotorSpeed, 'R:', rightMotorSpeed)
    writeToSerial(leftMotorSpeed, rightMotorSpeed, 1, 1)
    prev_error = error
    #integral_prior = integral
    #sleep(iteration_time)
'''
while(1):
    lms=100
    rms=200
    ser.write(b'f')
    print(lms, rms)
    #writeToSerial(100,200,1,1)
    #print(lms,rms)
    sleep(2)