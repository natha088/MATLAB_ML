%% MelanomaNet Training
% Run code from 'Combined' folder
%% Create Image Datastore with Labels
imds = imageDatastore(string(cd));
A = [categorical(repmat({'Benign'},[359 1]));categorical(repmat({'Malignant'},[283 1]))];
imds.Labels = A; 
%% Train Neural Network
net = alexnet; % load AlexNet

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.80,'randomized'); % 80% for Training, 20% Validation
augimdsTrain = augmentedImageDatastore([227 227],imdsTrain); % resize to 227x227
augimdsValidation = augmentedImageDatastore([227 227],imdsValidation); % resize to 227x227

layersTransfer = net.Layers(1:end-3); % remove three last layers 
numClasses = numel(categories(imdsTrain.Labels)); % number of classes (2)
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',5,'BiasLearnRateFactor',1);
    softmaxLayer
    classificationLayer]; % add three last layers of pre-trained network

% Set training options
options = trainingOptions('sgdm',... 
    'MiniBatchSize',22, ...
    'MaxEpochs',3, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',imdsValidation,...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Find accuracy of validation ran through network
netTransfer = trainNetwork(imdsTrain,layers,options); 

[YPred,probs] = classify(netTransfer,imdsValidation);
a = YPred == imdsValidation.Labels;
accuracy = mean(a);

%% Within-Data Error
[Pred_wdata,p_wdata] = classify(netTransfer,imds);
accuracy_data = mean(Pred_wdata == imds.Labels);

%% Specificity & Sensitivity

specificity = sum(Pred_wdata(1:359) == imds.Labels(1:359))/359;
sensitivity = sum(Pred_wdata(360:642) == imds.Labels(360:642))/283;

%% Save Network
MelanomaNet = netTransfer;
save MelanomaNet;


