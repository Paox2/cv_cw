clear all;
close all;  
%% Retrive training and testing images

dataPath='.\lfw\lfw\';

%% hp
hp.learning_rate = 0.0005;
hp.maxEpochs = 70;
hp.miniBatchSize = 128;


opts = trainingOptions("rmsprop",...
            "InitialLearnRate",hp.learning_rate,...
            'MaxEpochs',hp.maxEpochs,...
            'MiniBatchSize',hp.miniBatchSize,...
            'Plots','training-progress');


% options = trainingOptions('adam',...
%     'MiniBatchSize',128,...
%     'MaxEpochs',5,...
%     'InitialLearnRate',5e-3,...
%     'Verbose',1,...
%     'ExecutionEnvironment','cpu',...
%     'Plots','training-progress',...
%     'CheckpointPath', 'checkpoints/');
        
%% Load and Preprocess --- train data

nnInputSize = [100 100];
[trainImgSet, trainPersonID]=loadTrainingSet2(dataPath, 1);

detectedImgs = faceDetection(trainImgSet, []);
clear trainImgSet;

% data augmentation
trainSet = zeros(nnInputSize(1),nnInputSize(2),...
    size(detectedImgs,3),size(detectedImgs,4), 'uint8');
for i = 1 : size(detectedImgs,4)
    currImg = detectedImgs(:,:,:,i);
    trainSet(:,:,:,i) = imresize(currImg, nnInputSize);
end

trainLabel = trainPersonID;
trainLabel = trainLabel(:);

% get train and label set
yTrain = categorical(trainLabel);
xTrain = trainSet;

%% train model
nLabel = max(trainLabel);
lgraph = classifiarCNN.createNet(nLabel);

net = classifiarCNN.trainCNN(xTrain, yTrain, lgraph, opts);

clear xTrain
clear yTrain
clear opts
clear trainLabel
clear trainPersonID
clear detectedImgs
save net