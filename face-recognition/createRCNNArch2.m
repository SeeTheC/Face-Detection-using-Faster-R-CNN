% Creates the Faster R-CNN Architecture
function [layers,options,minInputDim] = createRCNNArch2(noOfClass,checkpointBasePath)    
    [layers,minInputDim] = createRCNNlayer(noOfClass);
    [options] = createRCNNOption(checkpointBasePath);
end

function [layers,minInputDim] = createRCNNlayer(noOfClass)
    minInputDim=[100 100 3];
    % Image input layer.
    inputLayer = imageInputLayer(minInputDim);
    % Define the convolutional layer parameters
    filterSize = [3 3];
    numFilters = 32;

    % Middle layers.
    middleLayers = [                
        convolution2dLayer(filterSize, numFilters, 'Padding', 1)   
        reluLayer()
        convolution2dLayer(filterSize, numFilters, 'Padding', 1)  
        reluLayer() 
        maxPooling2dLayer(3, 'Stride',2)       
     ];
 
    finalLayers = [    
        fullyConnectedLayer(64)
        reluLayer()     
        fullyConnectedLayer(noOfClass)
        softmaxLayer()
        classificationLayer()
     ];

    layers = [
        inputLayer
        middleLayers
        finalLayers
    ];

end

function [options] = createRCNNOption(checkpointBasePath)
    % Options for step 1.
    optionsStage1 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', checkpointBasePath);

    % Options for step 2.
    optionsStage2 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', checkpointBasePath);

    % Options for step 3.
    optionsStage3 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', checkpointBasePath);

    % Options for step 4.
    optionsStage4 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', checkpointBasePath);

    options = [
        optionsStage1
        optionsStage2
        optionsStage3
        optionsStage4
      ];

end