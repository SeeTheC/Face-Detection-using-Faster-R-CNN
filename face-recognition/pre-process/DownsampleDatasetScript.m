%% It will downsample the Dataset
%% INIT: Config

fprintf('Initializing..\n');
server = 1
timestamp=datestr(now,'dd-mm-yyyy HH:MM:SS');
fprintf('Timestampe: %s \n',timestamp);
if server 
    basepath = '~/git/Face-Recognition-using-Faster-R-CNN/data';
else
    basepath = '../data';
end
savedBasepath=strcat(basepath,'/Wider_',timestamp);
basepath = strcat(basepath,'/Wider');

maxHeight=200;
maxDim=[maxHeight,maxHeight];
fprintf('**Max Height Image: %d\n',maxHeight);
%% Init File Path
fprintf('Init "Wider dataset" filepath..\n');
dataPath{1,1} = strcat(basepath,'/wider_face_split','/parse_train_dataset.mat');
dataPath{2,1} = strcat(basepath,'/wider_face_split','/parse_val_dataset.mat');

% savepath
saveDatasetPath{1,1}=strcat(savedBasepath,'/WIDER_train/images');
saveDatasetPath{1,2}=strcat(savedBasepath,'/wider_face_split','/parse_train_dataset.mat');
saveDatasetPath{1,3}=strcat(savedBasepath,'/WIDER_train');

saveDatasetPath{2,1}=strcat(savedBasepath,'/WIDER_val/image');
saveDatasetPath{2,2}=strcat(savedBasepath,'/wider_face_split','/parse_val_dataset.mat');
saveDatasetPath{2,3}=strcat(savedBasepath,'/WIDER_val');

mkdir(strcat(savedBasepath,'/wider_face_split'));
for i=1:size(saveDatasetPath,1)
    mkdir(saveDatasetPath{i,1});
end
%% Downsample images
noOfDataSet=size(dataPath,1);
fprintf('Downsampling data.... \n');
for i=1:size(dataPath,1)
    fprintf('--------------------[Datapath:%d:]------------------\n',i);
    filepath=dataPath{i,1};      
    [dataset] = downsampeDataSet(dataPath{i,1},maxDim,savedBasepath,saveDatasetPath{i,3},basepath);
    save(saveDatasetPath{i,2},'dataset');
end

fprintf('Completed.... \n');
%%