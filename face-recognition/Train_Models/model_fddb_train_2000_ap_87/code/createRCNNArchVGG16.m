% Creates the Faster R-CNN Architecture
function [layers,options,minInputDim] = createRCNNArchVGG16(noOfClass,checkpointBasePath)  
    net=vgg16();
%    minInputDim=net.Layers(1).InputSize;
%    layersTransfer = net.Layers(1:end-3);
%    finalLayers = [    
%        fullyConnectedLayer(noOfClass,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%        softmaxLayer()
%        classificationLayer()
%     ];
%    layers=[
%            layersTransfer
%            finalLayers
%           ];
    layers=net;
    [options] = createRCNNOption(checkpointBasePath);
    minInputDim=250;
end

function [options] = createRCNNOption(checkpointBasePath)
    cpath=strcat(checkpointBasePath,'/checkpoint');
    mkdir(cpath);
    % Options for step 1.
    optionsStage1 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 1000, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', cpath);

    % Options for step 2.
    optionsStage2 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 1000, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', cpath);

    % Options for step 3.
    optionsStage3 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 1000, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', cpath);

    % Options for step 4.
    optionsStage4 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 1000, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', cpath);

    options = [
        optionsStage1
        optionsStage2
        optionsStage3
        optionsStage4
      ];

end
