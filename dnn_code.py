import os
import numpy as np
import keras
import cv2
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import random

tf.keras.applications.densenet.preprocess_inputmodel_v = tf.keras.applications.DenseNet201( include_top=False,weights="imagenet", input_tensor=None, input_shape=None,  pooling=None, classes=10,
)

from key import getMarkedImage
# load model
X = []
Y = []

base_path='dataset1/'
source_path=base_path 
for child in os.listdir(source_path):
    print(child)
    sub_path = os.path.join(source_path, child)
    if os.path.isdir(sub_path):
        for data_file in os.listdir(sub_path):
            features_train=getMarkedImage(os.path.join(sub_path, data_file))
            if(len(features_train)>0):
                features_train=features_train[:32]
                X.append(features_train)
                Y.append(child)
            print(len(features_train))
#print(X)
#print(Y)

from sklearn.preprocessing import LabelEncoder
labelBinarizer = LabelEncoder()
y = labelBinarizer.fit_transform(Y)
print(y)     

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), 
                                                    test_size=0.2, random_state=42)


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

dnnModel = models.Sequential()
dnnModel.add(layers.Dense(2000,activation="relu",input_shape=(32,)))
dnnModel.add(layers.Dense(1000,activation="relu"))
dnnModel.add(layers.Dense(600,activation="relu"))
dnnModel.add(layers.Dense(300,activation="relu"))
dnnModel.add(layers.Dense(200,activation="relu"))
dnnModel.add(layers.Dense(100,activation="relu"))
dnnModel.add(layers.Dense(50,activation="relu"))
dnnModel.add(layers.Dense(30,activation="relu"))
dnnModel.add(layers.Dense(10,activation="softmax"))
dnnModel.summary()



dnnModel.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
dnnModel.fit(X_train,y_train,epochs=1000,batch_size=300)

testloss, testAccuracy=dnnModel.evaluate(X_test,y_test)

print(testAccuracy)

dnnModel.save_weights('my_checkpoint')
dnnModel.load_weights('my_checkpoint')

X_test=getMarkedImage("dataset1/tree/File10.jpg")
#print([X_test])
y_pred=dnnModel.predict_classes([X_test])
print(y_pred)
