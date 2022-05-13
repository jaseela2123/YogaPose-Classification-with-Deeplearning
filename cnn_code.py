from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from PIL import Image


def create_cnn_model():
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=10, activation='softmax'))

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data Augmentation
    batch_size = 32
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       rotation_range=20,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('U:/train',
                                                     target_size=(256, 256),
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

    test_set = test_datagen.flow_from_directory('U:/YogaPose/gui/imageupload/test_set',
                                                target_size=(256, 256),
                                                batch_size=batch_size,
                                                class_mode='categorical')

    classifier.fit_generator(training_set,
                             steps_per_epoch=729 // batch_size,  # number of training set images, 729
                             epochs=100,   # Epoch number
                             validation_data=test_set,
                             validation_steps=229 // batch_size)  # number of test set images, 229

    classifier.save('my_model_multi_class10.h5')  # save model


def predict_cnn_model(predict_image):
    classifier = load_model('my_model_multi_class10.h5')  # load the model that was created using cnn_multiclass.py

    test_image = image.load_img(predict_image,
                                target_size=(256, 256))  # folder predictions with images that I want to test
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = ''

    result = classifier.predict(test_image)  # returns array

    if (result[0][0]) == 1:
        prediction = 'bridge'  # predictions in array are in alphabetical order
    elif result[0][1] == 1:
        prediction = 'childspose'
    elif result[0][2] == 1:
        prediction = 'downwarddog'
    elif result[0][3] == 1:
        prediction = 'mountain'
    elif result[0][4] == 1:
        prediction = 'plank'
    elif result[0][5] == 1:
        prediction = 'seatedforwardbend'
    elif result[0][6] == 1:
        prediction = 'tree'
    elif result[0][7] == 1:
        prediction = 'trianglepose'
    elif result[0][8] == 1:
        prediction = 'warrior1'
    elif result[0][9] == 1:
        prediction = 'warrior2'
    else :
        prediction = 'no pose matched'

    print(result)
    print(prediction)
    return prediction
