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
    basepath = './Train_Models/';
    if detectorType == DetectorType.WiderVgg16ImgDim600
        dtPath=strcat(basepath,'/model_vgg16_5epoch_600dim');
    elseif detectorType == DetectorType.WiderVgg16
        dtPath=strcat(basepath,'/model_all_vgg16_prec_18_80');    
    elseif detectorType == DetectorType.WiderAlexNet
        dtPath=strcat(basepath,'/model_all_alex_2_prec_14_76');
    elseif detectorType == DetectorType.FDDBVgg16Net
        dtPath=strcat(basepath,'/model_fddb_train_2000_ap_87');           
    end
    modelpath=strcat(dtPath,'/','detector.mat');
    sobj=load(modelpath);
    detector=sobj.detector; 
end

