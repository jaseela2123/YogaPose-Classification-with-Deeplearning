import cv2 
import numpy as np
import argparse
#import imutils
import time

def getMarkedImage(file_name):
    protoFile = "mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "mpi/pose_iter_160000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    frame = cv2.imread(file_name)

    inWidth = 500

    inHeight = 368

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]

    W = output.shape[3]

    points = []
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                       "Background": 15 }


    for i in range(len(BODY_PARTS)):

        probMap = output[0, i, :, :]

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > 0.1 :
            cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

            points.append((int(x), int(y)))

        else :

            points.append((0,0))
    str1=[]
    for e in points:
        for c in e:
            str1.append(c)
    print(str1)

    return str1