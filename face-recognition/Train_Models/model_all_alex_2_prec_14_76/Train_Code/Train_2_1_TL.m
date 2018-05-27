%% Face Recognitiong using Faster R-CNN: Train_Tl : Train using transfer learning
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
%basepath = strcat(basepath,'/Wider_MIN_16x16');
basepath = strcat(basepath,'/Wider');
%basepath = strcat(basepath,'/Wider_minH_400');
savedBasepath=strcat(basepath,'/trained_model');
savemodel=strcat(savedBasepath,'/model_',timestamp);

trainNewModel=false;
checkpointing=false;

if trainNewModel && checkpointing    
    savemodel=strcat(savedBasepath,'/model_all_alex_1');
    checkpointModelPath=strcat(savemodel,'/checkpoint/','/faster_rcnn_stage_4_checkpoint__9050__2018_04_25__21_17_47.mat');
end
if ~trainNewModel
    %savedModelPath=strcat(savedBasepath,'/train_200');
    %savedModelPath=strcat(savedBasepath,'/model_all_alex_1');
    savedModelPath=strcat(savedBasepath,'/model_all_alex_2_prec_20');        
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
fullTrainDataset=load(trainFile);
fullTrainDataset=fullTrainDataset.dataset;
fprintf('Completed..\n');
fullTrainDataset.Properties.VariableNames={'filename','box'};

%fullTrainDataset=fullTrainDataset(1:600,:);

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

%% Creating Arch using Trasfered Learning 
[layers,options,minInputDim]=createRCNNArchAlexNet(2,savemodel);
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
        'SmallestImageDimeion',600,...
        'BoxPyramidScale', 1.2);
    
    savepath=strcat(savemodel,'/','detector.mat');
    save(savepath,'detector');
    fprintf('Completed..\n');
else
    fprintf('Loading Pretrained Model..\n');
    % Loading Saved Model
    modelpath=strcat(savedModelPath,'/','detector.mat');
    load(modelpath,'detector');
    %%detector=sobj.detector; 
    fprintf('Completed..\n');
    
end
toc
%% TEST on one Image
fprintf('Testing on image..\n')
% Read a test image.
%I = imread(valData.filename{2});
%I = imread(valData.filename{200});
I = imread(trainingData.filename{200});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
if size(bboxes,1)>0
    I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
else
    fprintf('**NO FACE FOUND')
end
figure
imshow(I)
%%  Val Testing result
fprintf('\n-----------------[Val Testing PHASE]-------------------------------\n');
trainNewModel=false;savedModelPath=savemodel; %TEMP

if trainNewModel    
    [avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,minInputDim,valData,savemodel);    

else
    resultpath=strcat(savedModelPath,'/','test_result.mat');
    if  ~exist(resultpath)
        [avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,minInputDim,valData,savedModelPath);    
    else 
        prPath=strcat(savedModelPath,'/','precision_recall.mat');
        fprintf('Loading Pretrained Result..\n');
        % Loading Saved Model
        sobj=load(resultpath);
        result=sobj.result; 
        
        expectedResults = valData(:, 2:end);
        expectedResults(1501,:)=[];result(1501,:)=[];
        expectedResults(1555,:)=[];result(1555,:)=[];
        expectedResults(1819,:)=[];result(1819,:)=[];
        expectedResults(1851,:)=[];result(1851,:)=[];        
        expectedResults(3239,:)=[];result(3239,:)=[];                
        expectedResults(4650,:)=[];result(4650,:)=[];       
        [avgPrecision, recall, precision] = evaluateDetectionPrecision(result, expectedResults);    
        tblPrecsionRecall=table(precision,recall);
        
        %sobj=load(prPath);
        %tblPrecsionRecall=sobj.tbl;         
        fprintf('Completed..\n');
    end
end
tblPrecsionRecall
fprintf('**Avg Precision of Dectector:%f ',avgPrecision);
%% Plot of Precision and Recall
figure
plot(tblPrecsionRecall.recall,tblPrecsionRecall.precision)
grid on
title(sprintf('Average Precision = %.1f\n',avgPrecision))

%% TEST DATA
fprintf('\n-----------------[Testing PHASE]-------------------------------\n');
testFile= strcat(matPath,'/parse_val_dataset.mat');
sds=load(testFile);
fullTestDataset=sds.dataset;
fullTestDataset.Properties.VariableNames={'filename','box'};
fullTestDataset.filename = fullfile(basepath, fullTestDataset.filename);
if trainNewModel
    saveTestModel=strcat(savemodel,'/Test_Result');
else
    saveTestModel=strcat(savedModelPath,'/Test_Result');
end

[avgPrecision,result,tblPrecsionRecall] = predictOnTestDataset(detector,minInputDim,fullTestDataset,saveTestModel);    
tblPrecsionRecall
fprintf('**Avg Precision of Dectector:%f ',avgPrecision);


%% Plot of Precision and Recall
figure
plot(tblPrecsionRecall.recall,tblPrecsionRecall.precision)
grid on
title(sprintf('Average Precision = %.2f\n',avgPrecision))
