import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from key import getMarkedImage

import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
#tf.random.set_random_seed(42)

dnnModel=models.Sequential()
dnnModel.add(layers.Dense(2000,activation="relu",input_shape=(32,)))
#dnnModel.add(layers.Dense(2000,activation="relu",input_shape=(1536,)))
#dnnModel.add(layers.Dense(2000,activation="relu",input_shape=(18432,)))
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
dnnModel.load_weights('my_checkpoint')
X_test=getMarkedImage("dataset1/tree/File16.jpg")
#print([X_test])
y_pred=dnnModel.predict_classes([X_test])
print(y_pred)