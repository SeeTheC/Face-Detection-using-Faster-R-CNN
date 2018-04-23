%% Face Recognitiong using Faster R-CNN: Train
%% Init
fprintf('Initializing..\n');
server = 1
if server 
    basepath = '../data';
else
    basepath = '../data';
end
savedBasepath=strcat(basepath,'/savepath');

valPercent=0.4
%% Init Dataset

% Reading Wider dataset
fprintf('Init Wider dataset filepath..\n');
matPath = strcat(basepath,'/wider_face_split');
trainFile= strcat(matPath,'/parse_train_dataset.mat');
testFile= strcat(matPath,'/parse_val_dataset.mat');
fullTrainDataset=load(trainFile);
fullTrainDataset=fullTrainDataset.dataset;
fprintf('Completed..\n');
fullTrainDataset=fullTrainDataset(1:300,:);

% Display first few rows of the data set.
fullTrainDataset(1:4,:)
% Creating full path
fullTrainDataset.filename = fullfile(basepath, fullTrainDataset.filename);
% full path
fullTrainDataset(1:4,:)

%% Visualizing Dataset 
% Read one of the images.
I = imread(fullTrainDataset.filename{1});
I = insertShape(I, 'Rectangle', fullTrainDataset.box{1});
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
fprintf('Started training..\n');
train=true;
if train
    % Set random seed to ensure example training reproducibility.
    rng(0);    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.65 1], ...
        'BoxPyramidScale', 1.2);
else
    % Load pretrained detector for the example.
    
end
fprintf('Completed..\n');
 