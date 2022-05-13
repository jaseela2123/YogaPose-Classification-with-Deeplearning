### key.py
#-----------

import cv2 
import numpy as np

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
  
  
  
### SVM, RF, KNN
#-----------------

import os
import numpy as np
import keras
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from key import getMarkedImage

# load model
X = []
Y = []

base_path='dataset1/'
source_path=base_path 
for child in os.listdir(source_path):
    print(child)
    sub_path = os.path.join(source_path, child)
    bsub_path = os.path.join(base_path, child)
    if os.path.isdir(sub_path):
        for data_file in os.listdir(sub_path):
            features_train = getMarkedImage(os.path.join(sub_path, data_file))
            if(len(features_train)>0):
                features_train=features_train[:32]
                X.append(features_train)
                Y.append(child)
            print(len(features_train))

from sklearn.preprocessing import LabelEncoder
labelBinarizer = LabelEncoder()
y = labelBinarizer.fit_transform(Y)
print(y)     

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), 
                                                    test_size=0.2, random_state=42)

res = [getMarkedImage('E:/open_pose/code/dataset1/downwarddog/File6.jpg')]


### 1) RandomForest

from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def RandomForest(X_train,y_train):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.predict(res[:]))
    cm=confusion_matrix(y_test, y_pred)
    print(cm)
    df_cm = pd.DataFrame(cm, index = labelBinarizer.classes_, columns =labelBinarizer.classes_ )
    plt.figure(figsize = (10,7))
    plt.title("Confusion Matrix -  RandomForestClassifier")
    sn.heatmap(df_cm, annot=True,vmin=-20, vmax=40,cmap="Blues")
    print(classification_report(y_test, y_pred))
  

RandomForest(X_train,y_train)



### 2) SVC

from sklearn.svm import SVC

clf = SVC(C=0.7, gamma='scale', kernel='rbf', probability=True)
clf.fit(X_train,y_train)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

print(res)

y_pred = clf.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)
df_cm = pd.DataFrame(cm, index = labelBinarizer.classes_, columns =labelBinarizer.classes_ )
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,vmin=-20, vmax=40,cmap="Blues")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print(clf.predict(res[:]))


### 3) KNN

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.40, random_state=42) 

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train.ravel())

y_pred = classifier.predict(X_test)
print(classifier.score(X_test,y_test))
predicted= classifier.predict(res)
if predicted==1:
  print("Bridge")
elif predicted == 2 :
  print("Child")
elif predicted == 3 :
  print("downwarddog")
elif predicted == 4 :
  print("mountain")
elif predicted == 5 :
  print("plank")
elif predicted == 6 :
  print("seatedforwardbend")
elif predicted == 7 :
  print("tree")
elif predicted == 8 :
  print("trianglepose")
elif predicted == 9 :
  print("warrior1")
elif predicted == 10 :
  print("warrior2")
else:
  print("nomatch")
