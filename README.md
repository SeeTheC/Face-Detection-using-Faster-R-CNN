# Face Detection using Faster-R-CNN
Face Recognition using Faster R-CNN

## Abstract
Face detection has vast applications in the areas ranging from surveillance, security, crowd size estimation to social networking etc. The challenge lies in creating a model which is agnostic to lightning conditions, pose, accessories and occlusion. We aim to create a pipeline which takes an image as an input and creates a bounding box on the faces of all the people in the image. 

## Dependencies
- Matlab 2018
  - [Neural Network Toolbox](https://www.mathworks.com/products/neural-network.html)
  - [AlexNet](https://www.mathworks.com/help/nnet/ref/alexnet.html)
  - [VGG16](https://www.mathworks.com/help/nnet/ref/vgg16.html)

## Instructions for running the code
- Pre-Processing
  - For FDDB RUN [parseFddbDataset.m](face-recognition/parseFddbDataset.m)
  - For WIDER RUN [parseWiderDataset.m](face-recognition/parseWiderDataset.m) 
- Training and testing 
  - For WIDER dataset on VGG16 Run [trainWIDER.m]([face-recognition/Train.m)
  - For FDDB dataset on VGG16 Run [trainFDDB.m](face-recognition/Ayush_Train_VGG_FDDB.m)
  
For training your own model:
 - Write you own Architecture function(specifying model and its config).
    - smaple Architecture fuctions are [createRCNNArchVGG16.m](face-recognition/createRCNNArchVGG16.m), [createRCNNArchAlexNet.m](face-recognition/createRCNNArchAlexNet.m)
 - Change the architecture function name in train file 
    - For WIDER Dataset [[trainWIDER.m](face-recognition/Train_2_1_TL.m)
    - For FDDB dataset  [trainFDDB.m](face-recognition/Ayush_Train_VGG_FDDB.m)
 - You can also change the train and test data percent in [trainFDDB.m](face-recognition/Ayush_Train_VGG_FDDB.m) for FDDB dataset. 
 - WIDER has predefined seperated train and test image dataset.

## Overall Details
- We used following dataset:
  - WIDER
    -32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images.WIDER FACE dataset is organized based on 61 event classes. For each event class, we randomly select 40%/10%/50% data as training, validation and testing sets.
  - FDDB. This data set contains the annotations for 5171 faces in a set of 2845 images.
  - Some of our own images

- We Train our dataset on following model:
  - Alexnet
  - VGG16
  - Made our own model on 11 layers

- We had following constraints:
  - Due to lack of gpu we have to reduce size of our images.
  example: WIDER images had (1200X1500) approx size. We have to reduce size of the image to (600X(600*rescaleFactor)).
  - Due to reduction in size of images some of the faces which were almost not visible (as small as a dot). Detector ignored these faces.
  - This reduced the accuracy of the model. Otherwise it would have shown overall better accuracy.
  - Traing process was time consuming because of lack of resources.



- Training and Testing Data
  - FDDB on VGG16
    - Trained Data - 60%
    - Test Data - 40%
    - MAP estimate - 87%
  - WIDER on VGG16
    - Trained Data With reduced Dimension of images- 40%
    - Test Data on- 50%
    - MAP estimate - 20% - 25% (with many of the faces getting ignored due to reduced dimension)
  - WIDER on Alexnet
    - Trained Data With reduced Dimension of images- 40%
    - Test Data on- 50%
    - MAP estimate - 14% - 18% (with many of the faces getting ignored due to reduced dimension)
  - WIDER on our own model(13 layers)
    - Train data - 1000 images
    - Testing data - 300 images
    - MAP estimate - 35%
- Detection Time
  - It took 0.006 secs to detect an image of dimension (700X1000) with 10-15 faces. 

## Some observations  
  - WIDER and FDDB on VGG16 worked very well
  - WIDER and FDDB on Alexnet give less accuracy than VGG16
  - We tested VGG16 model trained on WIDER and tested ON FDDB. IT gave 82% accuracy.
  - We tested Alexnet model trained on WIDER and tested ON FDDB. IT gave 79.4% accuracy.
  - On our own model with 11 layers training took lot of time. Because of less number of "Max pooling" layer
  - We tested our own images on VGG16 trained over WIDER network and it worked really well!
  - WIDER trained model doesn't show good MAP estimate on WIDER test images because many faces get ignored due to reduce dimension of images. It shows good MAP estimate when WIDER trained VGG16 model is tested on FDDB(because no dimension reduction here on FDDB dataset).
  
## Here are the plots of the losses and accuracies of some of the best models
- Graph showing MAP estimate on FDDB dataset trained over VGG16
    - ![Sample 5](samples/precision_graph_FDDB.png)
- Graph showing MAP estimate on WIDER dataset trained over VGG16
    - ![Sample 6](samples/accuracy_WIDER.png)
- Graph showing MAP estimate on FDDB dataset tested over VGG16 model which is trained over WIDER.
    - ![Sample 7](samples/ACCURACyOfFDDBonWIDErTRAInMODEL.png)
 

## Results
### Test image of our own Dataset
- ![Sample 1](samples/cs2016-19.png)
- ![Sample 2](samples/csera-19.png)
### Test image of WIDER dataset 
- ![Sample 3](samples/29_Students_Schoolkids_Students_Schoolkids_29_251.jpg)
- ![Sample 4](samples/10_People_Marching_People_Marching_2_373.jpg)
    
 
 
 ## References
 - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
 - [Face Detection Data Set and Benchmark Home](http://vis-www.cs.umass.edu/fddb/) 
 - [WIDER FACE: A Face Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
