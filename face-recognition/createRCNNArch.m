% Creates the Faster R-CNN Architecture
function [layers,options] = createRCNNArch(noOfClass)    
    [layers] = createRCNNlayer(noOfClass);
    [options] = createRCNNOption();
end

function [layers] = createRCNNlayer(noOfClass)
    % Image input layer.
    inputLayer = imageInputLayer([32 32 3]);
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
        % Add a fully connected layer with 64 output neurons. The output size
        % of this layer will be an array with a length of 64.
        fullyConnectedLayer(64)
        
        % Add a ReLU non-linearity.
        reluLayer()

        % Add the last fully connected layer. At this point, the network must
        % produce outputs that can be used to measure whether the input image
        % belongs to one of the object classes or background. This measurement
        % is made using the subsequent loss layers.
        fullyConnectedLayer(noOfClass)

        % Add the softmax loss layer and classification layer. 
        softmaxLayer()
        classificationLayer()
     ];

    layers = [
        inputLayer
        middleLayers
        finalLayers
    ];

end

function [options] = createRCNNOption()
    % Options for step 1.
    optionsStage1 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir);

    % Options for step 2.
    optionsStage2 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir);

    % Options for step 3.
    optionsStage3 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir);

    % Options for step 4.
    optionsStage4 = trainingOptions('sgdm', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'CheckpointPath', tempdir);

    options = [
        optionsStage1
        optionsStage2
        optionsStage3
        optionsStage4
      ];

end