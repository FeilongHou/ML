import os
import zipfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import random
from skimage import transform
from skimage import io
from skimage import exposure

"""
zip_ref = zipfile.ZipFile('archive.zip', 'r')
zip_ref.extractall('archive')

data_points = open('archive/Train.csv').read().split("\n")
print(data_points[1:10])
"""

def load_data(folderPath, csvPath):
    # initialize picture and labels list
    picture = []
    labels = []
    # open the csv file and read and split according to \n    split() will split str into array base on \n
    # [1:] disregard the first line since they are names 
    # Width	Height	Roi.X1	Roi.Y1	Roi.X2	Roi.Y2	ClassId	Path
    data_points = open(csvPath).read().strip().split("\n")[1:]

    # all data points are sorted, we need to shuffle them
    random.shuffle(data_points)

    # now loop throught all the points
    for point in data_points:

        # getting label and picture path for each data point
        (label, picture_path) = point.strip().split(",")[-2:]

        # get the full path from the disk
        # sep will add / in path
        picture_path = os.path.sep.join([folderPath, picture_path])
        # read in the image
        image = io.imread(picture_path)

        image = transform.resize(image, (32,32))
        # adding exposure
        image = exposure.equalize_adapthist(image, clip_limit=0.1)


        
        picture.append(image)
        labels.append(label)
    
    picture = np.array(picture)
    labels = np.array(labels)

    return (picture, labels)


#****************************
# constructing neural network
#****************************

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (5,5), padding="same", activation = 'relu', input_shape = (32,32,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3),  activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(43, activation = 'softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

"""
x_train, y_train = load_data('archive', 'archive/Train.csv')
x_val, y_val = load_data('archive', 'archive/Test.csv')

print(x_train[10], y_train[10])
# reduce dimension value by 255
x_train = x_train / 255.0
x_val = x_val / 255.0

num_labels = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_labels)
y_val = tf.keras.utils.to_categorical(y_val, num_labels)

# originally had 30 epochs, but it it is not necessary
model.fit(x_train, y_train, batch_size = 64, validation_data = (x_val, y_val), epochs = 20)
model.save('Traffic')
"""
model = tf.keras.models.load_model("Traffic")

img = image.load_img('archive/Test/00217.png', target_size=(32, 32))
#img = transform.resize(img, (32,32))

x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
prediction = model.predict(images, batch_size = 10)

print(np.argmax(prediction[0]))
