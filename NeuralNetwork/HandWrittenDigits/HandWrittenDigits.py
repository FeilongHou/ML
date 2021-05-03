import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

training = open("train-images.idx3-ubyte", 'rb')
trianingL = open("train-labels.idx1-ubyte", 'rb')
testing = open("t10k-images.idx3-ubyte", 'rb')
testingL = open("t10k-labels.idx1-ubyte", 'rb')

image_size = 28
num_images = 60000
training.read(16)
buf = training.read(image_size * image_size * num_images)
train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train = train.reshape(num_images, image_size, image_size, 1)
train_images = np.asarray(train).squeeze()

TEST_NUMBER = 10000
testing.read(16)
buf = testing.read(image_size * image_size * TEST_NUMBER)
test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test = test.reshape(TEST_NUMBER, image_size, image_size, 1)
test_images = np.asarray(test).squeeze()

trianingL.read(8)
testingL.read(8)
train_labels = np.zeros(num_images)
test_labels = np.zeros(TEST_NUMBER)
for i in range(num_images):
    buf = trianingL.read(1)
    train_labels[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

for j in range(TEST_NUMBER):
    buf = testingL.read(1)
    test_labels[j] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (image_size,image_size)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 5)
model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)


for x in range(300,310,1):
    plt.grid = False
    plt.imshow(test_images[x])
    plt.xlabel("Actual = " + str(test_labels[x]))
    plt.title("Prediction = " + str(np.argmax(prediction[x])))
    plt.show()

"""
for x in range(TEST_NUMBER):
    plt.grid = False
    if np.argmax(prediction[x]) != test_labels[x]:
        plt.imshow(test_images[x], cmap = plt.cm.binary)
        plt.xlabel("Actual = " + str(test_labels[x]))
        plt.title("Prediction = " + str(np.argmax(prediction[x])))
        plt.show()"""