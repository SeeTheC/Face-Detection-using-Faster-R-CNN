% Img: Input image
% DetectorType: used for detecting the image
function [bboxImg] = faceDetector(img,detectorType)
    %% Load Detector
    detector=getDetector(detectorType);    
    %% detection face
    [bboxes,scores] = detect(detector,img);
    %% Marking BBox 
    if size(bboxes,1)>0
        bboxImg = insertObjectAnnotation(img,'rectangle',bboxes,scores);
    else
        fprintf('Warning:NO FACE FOUND');
        bboxImg=img;
    end
end
% Load the trained face detector
function [detector]=getDetector(detectorType)
    basepath = '~/git/Face-Recognition-using-Faster-R-CNN/data';
    basepath=strcat(basepath,'/Wider/trained_model');
    if detectorType == DetectorType.WiderVgg16
        dtPath=strcat(basepath,'/model_vgg16_5epoch_600dim');
    elseif detectorType == DetectorType.WiderAlexNet
        dtPath=strcat(basepath,'/model_vgg16_5epoch_600dim');
    elseif detectorType == DetectorType.FDDBAlexNet
        dtPath=strcat(basepath,'/model_vgg16_5epoch_600dim');           
    end
    modelpath=strcat(dtPath,'/','detector.mat');
    sobj=load(modelpath);
    detector=sobj.detector; 
end

