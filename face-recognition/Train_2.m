%% Face Recognitiong using Faster R-CNN: Train
%% Init: Config
clear all;
%----------------------------[Config]--------------------------------------
fprintf('Initializing..\n');
server = 1
if server 
    basepath = '~/git/Face-Recognition-using-Faster-R-CNN/data';
else
    basepath = '../data';
end
timestamp=datestr(now,'dd-mm-yyyy HH:MM:SS');
basepath = strcat(basepath,'/Wider_MIN_16x16');
savedBasepath=strcat(basepath,'/trained_model');
savemodel=strcat(savedBasepath,'/model_',timestamp);

trainNewModel=false;
if ~trainNewModel
    savedModelPath=strcat(savedBasepath,'/train_200');
end
valPercent=0.4;
%-------------------------------------------------------------------------

% Creatining dir
mkdir(savedBasepath);
mkdir(savemodel);
%% Init Dataset

% Reading Wider dataset
fprintf('Init Wider dataset filepath..\n');
matPath = strcat(basepath,'/wider_face_split');
trainFile= strcat(matPath,'/parse_train_dataset.mat');
testFile= strcat(matPath,'/parse_val_dataset.mat');
fullTrainDataset=load(trainFile);
fullTrainDataset=fullTrainDataset.dataset;
fprintf('Completed..\n');
fullTrainDataset.Properties.VariableNames={'filename','box'};

fullTrainDataset=fullTrainDataset(1:200,:);

% Display first few rows of the data set.
fullTrainDataset(1:2,:)
% Creating full path
fullTrainDataset.filename = fullfile(basepath, fullTrainDataset.filename);
% full path
fullTrainDataset(1:2,:)

%% Visualizing Dataset 
% Read one of the images.
imgNo=2;
I = imread(fullTrainDataset.filename{imgNo});
I = insertShape(I, 'Rectangle', fullTrainDataset.box{imgNo});
figure
imshow(I);
%% Creating Training anf validation Set
idx = floor((1-valPercent) * height(fullTrainDataset));
trainingData = fullTrainDataset(1:idx,:);
testData = fullTrainDataset(idx:end,:);
%% Creating Arch
[layers,options]=createRCNNArch(2);
layers
%% Train

if trainNewModel
    fprintf('Started training..\n');
    rng(0);    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.65 1], ...
        'BoxPyramidScale', 1.2);
    
    savepath=strcat(savemodel,'/','detector.mat');
    save(savepath,'detector');
    fprintf('Completed..\n');
else
    fprintf('Loading Pretrained Model..\n');
    % Loading Saved Model
    modelpath=strcat(savedModelPath,'/','detector.mat');
    sobj=load(modelpath);
    detector=sobj.detector; 
    fprintf('Completed..\n');
    
end
%% TEST on one Image
fprintf('Testing on image..\n')
% Read a test image.
I = imread(testData.filename{1});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
%%  Testing result
fprintf('\n-----------------[Testing PHASE]-------------------------------\n');
resultpath=strcat(savedModelPath,'/','test_result.mat');
if trainNewModel
    
    [avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,testData,savemodel);    

elseif (~trainNewModel && ~exist(resultpath))    

    [avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,testData,savedModelPath);    

else 
    prPath=strcat(savedModelPath,'/','precision_recall.mat');
    fprintf('Loading Pretrained Result..\n');
    % Loading Saved Model
    sobj=load(resultpath);
    result=sobj.result; 
    sobj=load(prPath);
    tblPrecsionRecall=sobj.tbl;         
    fprintf('Completed..\n');    
end
tblPrecsionRecall
fprintf('**Avg Precision of Dectector:%f ',avgPrecision);
%% Plot of Precision and Recall
figure
plot(tblPrecsionRecall.recall,tblPrecsionRecall.precision)
grid on
title(sprintf('Average Precision = %.1f\n',avgPrecision))


 