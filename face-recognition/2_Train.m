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

%% Init Dataset

% Reading Wider dataset
fprintf('Init Wider dataset filepath..\n');
matPath = strcat(basepath,'/wider_face_split');
trainFile= strcat(matPath,'/parse_train_dataset.mat');
testFile= strcat(matPath,'/parse_val_dataset.mat');
trainDataset=load(trainFile);
trainDataset=trainDataset.dataset;
fprintf('Completed..\n');
%%
