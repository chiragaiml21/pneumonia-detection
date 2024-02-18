import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
# import matplotlib.pyplot as plt

class_names = ["NORMAL", "PNEUMONIA"]

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', activation='relu', input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.load_weights("models/pneumonia_detection_grayscale.h5")

def predict_image(file_path):
    img = load_img(file_path, target_size=(256, 256), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    # result = np.argmax(predictions)

    # print(predictions)
    if(predictions[0][0]>0.3):
      result = 1
    else:
      result = 0

    print(f'Predicted : {class_names[result]}')
    # plt.imshow(img)
    # plt.title(f'Predicted: {class_names[result]}')
    # plt.axis('off')
    # plt.show()

image_path = 'prediction_data/NORMAL2-IM-1440-0001.jpeg'
predict_image(image_path)