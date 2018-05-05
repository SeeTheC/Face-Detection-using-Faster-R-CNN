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
basepath = strcat(basepath,'/FDDB');
savedBasepath=strcat(basepath,'/trained_model');
savemodel=strcat(savedBasepath,'/model_',timestamp);

trainNewModel=false;
checkpointing=false;

if trainNewModel && checkpointing
    checkpointModelPath=strcat(basepath,'/trained_model/checkpoint/','/faster_rcnn_stage_2_checkpoint__50892__2018_04_25__04_20_10.mat');
end
if ~trainNewModel
    savedModelPath=strcat(savedBasepath,'/model_train_2000_ap_87');
end
valPercent=0.2;
%-------------------------------------------------------------------------

% Creatining dir
mkdir(savedBasepath);
mkdir(savemodel);
%% Init Dataset

% Reading FDDB dataset
fprintf('Init FDDB dataset filepath..\n');
matPath = strcat(basepath,'/FDDB-folds');
trainFile= strcat(matPath,'/parseFDDB_dataset.mat');
%testFile= strcat(matPath,'/parse_val_dataset.mat');
fullTrainDataset=load(trainFile);
fullTrainDataset=fullTrainDataset.finaltbl;
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
        'SmallestImageDimension', 250, ...
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
%% TEST on one Image
fprintf('Testing on image..\n')
% Read a test image.
I = imread(valData.filename{2});
%I = imread(trainingData.filename{1});

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
%%  Test
doTrainingAndEval=1;
testData=valData;
minInputDim=[250,250];
if doTrainingAndEval
    % Run detector on each image in the test set and collect results.
    resultsStruct = struct([]);
    for i = 1:height(testData)
        
        % Read the image.
        I = imread(testData.filename{i});
        if(size(I,1)<minInputDim(1) || size(I,2)<minInputDim(2))
            fprintf('Error:Dim of Image is less than required [%d,%d] dim:[%d,%d]\n',minInputDim(1),minInputDim(2),size(I,1),size(I,2))
            continue;
        end 
        % Run the detector.
        [bboxes, scores, labels] = detect(detector, I);
        
        % Collect the results.
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;
    end
    
    % Convert the results into a table.
    results = struct2table(resultsStruct);
else
    % Load results from disk.
    results = data.results;
end

% Extract expected bounding box locations from test data.
expectedResults = testData(:, 2:end);

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults)
%%
% Plot precision/recall curve
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))


%%  Testing result
fprintf('\n-----------------[Testing PHASE]-------------------------------\n');
minInputDim=[250,250];
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
        sobj=load(prPath);
        tblPrecsionRecall=sobj.tbl;         
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


 
