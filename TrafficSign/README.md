# AI Identify Traffic Sign

This project is done using python and TensorFlow CNN

## Here are examples of images:
Training Images:\
![Sample1](https://github.com/FeilongHou/ML/blob/main/TrafficSign/00003_00000_00000.png)\
![Sample2](https://github.com/FeilongHou/ML/blob/main/TrafficSign/00003_00000_00029.png)\
![Sample3](https://github.com/FeilongHou/ML/blob/main/TrafficSign/00003_00001_00029.png)\
Testing Images:\
![Test1](https://github.com/FeilongHou/ML/blob/main/TrafficSign/3.png)\
![Test2](https://github.com/FeilongHou/ML/blob/main/TrafficSign/14.png)\
Training Images are ranging from 28x28 to 189x189 colored pictures with different exposure and clarity. Thus image resize to 32x32 is needed for 2 convolutional layer 2 dense layer neural network to classify images.\
Validation set of the same sized images are used to increase accuracy.\

This AI originally is only able to achieve 80% validation accuracy and 65% prediction accuracy.\
Changing image resieze to 64x64 and adding additional convolutional layer result in 10% increase in prediction accuracy.\

**All Images Are From The German Traffic Sign Recognition Benchmark (GTSRB)**
