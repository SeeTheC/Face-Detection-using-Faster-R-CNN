%% Face Recognitiong using Faster R-CNN: Train_Tl : Train using transfer learning
%% Init: Config
clear all;
%%
%----------------------------[Config]--------------------------------------
fprintf('Initializing..\n');
server = 1
if server 
    basepath = '~/git/Face-Recognition-using-Faster-R-CNN/data';
else
    basepath = '../data';
end
timestamp=datestr(now,'dd-mm-yyyy HH:MM:SS');
%basepath = strcat(basepath,'/Wider_MIN_16x16');
basepath = strcat(basepath,'/Wider');
savedBasepath=strcat(basepath,'/trained_model');
savemodel=strcat(savedBasepath,'/model_',timestamp);

trainNewModel=true;
checkpointing=false;

if trainNewModel && checkpointing
    checkpointModelPath=strcat(basepath,'/trained_model/checkpoint/','/faster_rcnn_stage_3_checkpoint__23176__2018_04_27__06_17_39.mat');
end
if ~trainNewModel
    savedModelPath=strcat(savedBasepath,'/model_all_VGG16_5epochforall');
end
valPercent=0;
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

%fullTrainDataset=fullTrainDataset(1:300,:);

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
valData = fullTrainDataset(idx:end,:);
fprintf('Division completed..\n');
%% Creating Arch using Trasfered Learning 
[layers,options,minInputDim]=createRCNNArchVGG16(2,savemodel);
if checkpointing
    data=load(checkpointModelPath);
    layers=data.detector;
end
layers
%% Train
tic
if trainNewModel
    fprintf('Started training..\n');
    rng(0);    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.    
    [detector,info] = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.65 1], ... 
        'SmallestImageDimension', 400, ...
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
toc
% %% TEST on one Image
% fprintf('Testing on image..\n')
% % Read a test image.
% I = imread(valData.filename{2});
% %I = imread(trainingData.filename{1});
% 
% % Run the detector.
% [bboxes,scores] = detect(detector,I);
% 
% % Annotate detections in the image.
% if size(bboxes,1)>0
%     I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% else
%     fprintf('**NO FACE FOUND')
% end
% figure
% imshow(I)

%% TEST DATA
fprintf('\n-----------------[Testing PHASE]-------------------------------\n');
FDDB=false;
if FDDB
    fddbBP='~/git/Face-Recognition-using-Faster-R-CNN/data/FDDB';     
    testFile= strcat(fddbBP,'/FDDB-folds','/parse_FDDB_dataset.mat');
    sds=load(testFile);
    fullTestDataset=sds.finaltbl;    
else
    testFile= strcat(matPath,'/parse_val_dataset.mat');
    sds=load(testFile);
    fullTestDataset=sds.dataset;    
end
fullTestDataset.Properties.VariableNames={'filename','box'};
if FDDB
     fullTestDataset.filename = fullfile(fddbBP, fullTestDataset.filename);
     %saveTestModel=strcat(fddbBP,'/FDDB');
     saveTestModel=strcat(savedModelPath,'/Test_Result'); 
elseif trainNewModel
    fullTestDataset.filename = fullfile(basepath, fullTestDataset.filename);
    saveTestModel=strcat(savemodel,'/Test_Result');
else
   fullTestDataset.filename = fullfile(basepath, fullTestDataset.filename);
   saveTestModel=strcat(savedModelPath,'/Test_Result');    
end
minInputDim=[240,240];
[avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,minInputDim,fullTestDataset,saveTestModel);    
tblPrecsionRecall
fprintf('**Avg Precision of Dectector:%f ',avgPrecision);
%% Plot of Precision and Recall
figure
plot(tblPrecsionRecall.recall,tblPrecsionRecall.precision)
grid on
title(sprintf('Average Precision = %.3f\n',avgPrecision))
xlabel('Recall');
ylabel('Precision');
 
