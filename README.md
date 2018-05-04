# Face-Recognition-using-Faster-R-CNN
Face Recognition using Faster R-CNN

## Abstract
Face detection has vast applications in the areas ranging from surveillance, security, crowd size estimation to social networking etc. The challenge lies in creating a model which is agnostic to lightning conditions, pose, accessories and occlusion. We aim to create a pipeline which takes an image as an input and creates a bounding box on the faces of all the people in the image. 

## Dependencies
- Matlab 2018
  - [Neural Network Toolbox](https://www.mathworks.com/products/neural-network.html)
  - [AlexNet](https://www.mathworks.com/help/nnet/ref/alexnet.html)
  - [VGG16](https://www.mathworks.com/help/nnet/ref/vgg16.html)

## Instructions for running the code
- Training and testing WIDER dataset on VGG16
  - Run face-recognition/Train_2_1_TL.m
- Training and testing FDDB dataset on VGG16
  - Run face-recognition/Ayush_Train_VGG_FDDB.m

## Results
- ![Sample 1](samples/cs2016-19.png)
- ![Sample 1](samples/csra-19.png)
- ![Sample 2](samples/29_Students_Schoolkids_Students_Schoolkids_29_251.jpg)
- ![Sample 3](samples/10_People_Marching_People_Marching_2_373.jpg)
