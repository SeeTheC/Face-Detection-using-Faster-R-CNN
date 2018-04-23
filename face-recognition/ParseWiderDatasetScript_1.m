%% Face Recognitiong using Faster R-CNN: Dataset parser
%% Init
fprintf('Initializing..\n');
server = 1
if server 
    basepath = '../data';
else
    basepath = '../data';
end

%% Init Dataset: Wider File Path

fprintf('Init Wider dataset filepath..\n');
matPath = strcat(basepath,'/wider_face_split');
trainFile= strcat(matPath,'/wider_face_train.mat');
valFile= strcat(matPath,'/wider_face_val.mat');
testFile= strcat(matPath,'/wider_face_test.mat');

savepath=strcat(basepath,'/wider_face_split');
%% Parsing Datset and Save
fprintf('Parsing dataset..\n');
[dataset] = parseWiderDataset(trainFile,'WIDER_train/images');
save(strcat(savepath,'/parse_train_dataset.mat'),'dataset');
[dataset] = parseWiderDataset(valFile,'WIDER_val/images');
save(strcat(savepath,'/parse_val_dataset.mat'),'dataset');
fprintf(' Completed \n');
%%
