import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

"""
# unzip file
zip_ref = zipfile.ZipFile('horse-or-human.zip', 'r')
zip_ref.extractall('horse-or-human')
zip_ref = zipfile.ZipFile('validation-horse-or-human.zip', 'r')
zip_ref.extractall('validation-horse-or-human')

# visualization 
train_horse_dir = os.path.join('horse-or-human/horses')
train_human_dir = os.path.join("horse-or-human/humans")
validation_horse_dir = os.path.join('validation-horse-or-human/horses')
validation_human_dir = os.path.join('validation-horse-or-human/humans')
train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))

rows = 4
cols = 4
pic_index = 0

# get current figure
fig = plt.gcf()
pic_index += 8

next_horse_pix = [os.path.join(train_horse_dir, name) 
                for name in train_horse_names[pic_index - 8 : pic_index]]

next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index - 8 : pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(rows, cols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
"""

"""
model = tf.keras.Sequential([
    # convolution layer: 16 convolution matrix, each is 3x3, input picture are 300x300 pixel with RGB
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (300, 300, 3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),  # drop some neurons for faster computation
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'tanh')
    ])


model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = 0.001), metrics = ['accuracy'])

# Rescale image by 1./255
train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)

# Flow training images in batches of 128
training = train_datagen.flow_from_directory('horse-or-human', target_size = (300,300), batch_size = 128, class_mode = 'binary')
validation = validation_datagen.flow_from_directory('validation-horse-or-human', target_size = (300,300), batch_size = 128, class_mode = 'binary')

model.fit(training, validation_data = validation, epochs = 15)

model.save('HoH2')
"""

model = tf.keras.models.load_model("HoH")

img = image.load_img('horse1.jpg', target_size=(300, 300))

x = image.img_to_array(img)
x = x / 255
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)

print(classes[0])

if classes[0]>0.5:
    print("horse1.jpg" + " is a human")
else:
    print("horse1.jpg" + " is a horse")
