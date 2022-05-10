clear;
close all;
clc;
%% Retrive training and testing images

dataPath='.\lfw\lfw\';

        
%% Load and Preprocess --- train data

nnInputSize = [100 100];
[trainImgSet, trainPersonID]=loadTrainingSet2(dataPath,0);

detectedImgs = faceDetection(trainImgSet, []);
clear trainImgSet;

% data augmentation
nExtend = 0;
nLabel = size(trainPersonID, 1);
trainSet = zeros(nnInputSize(1),nnInputSize(2),...
    size(detectedImgs,3),size(detectedImgs,4)*(nExtend+1), 'uint8');
for i = 1 : size(detectedImgs,4)
%     newI = [];
    currImg = detectedImgs(:,:,:,i);
    trainSet(:,:,:,i) = imresize(currImg, nnInputSize);
%     newI(:,:,:,1) = imresize(currImg, nnInputSize);
%         
%     xRel = randomAffine2d("XReflection", true);
%     relImg = imwarp(currImg, xRel);
%     newI(:,:,:,2) = imresize(relImg, nnInputSize);
%     
%     reI = imresize(currImg, nnInputSize);
%     reI(reI > 225) = 225;
%     lightI = reI + 30;
%     newI(:,:,:,3) = lightI;
%             
%     reI = imresize(currImg, nnInputSize);
%     reI(reI < 30) = 30;
%     lightI = reI - 30;
%     newI(:,:,:,4) = lightI;
%         
%     lightImg = jitterColorHSV(currImg,'Contrast',0.4,'Hue',0.1,'Saturation',0.2,'Brightness',0.3);
%     newI(:,:,:,5) = imresize(lightImg, nnInputSize);
        
%     for j = 0 : 4
%         trainSet(:,:,:,(nExtend+1)*i-j) = newI(:,:,:,j+1);
%     end
end

meanFace = trainSet - mean(trainset,4);

trainLabel = trainPersonID;
for i = 1 : nExtend
    trainLabel = [trainLabel;trainPersonID];
end
trainLabel = trainLabel(:);

% get train and label set
yTrain = categorical(trainLabel);
xTrain = trainSet;

%% clear
clear accuracy;
clear accuracyBatchSize;
clear ans;
clear currImg;
clear detectedImgs
clear executionEnvironment
clear faceImgTest
clear hp
clear i
clear I
clear index 
clear inputH
clear inputW
clear j
clear lightI
clear lightImg
clear loss
clear newI
clear nExtend
clear nLabel
clear nnInputSize
clear option
clear opts
clear pairLabelsAcc
clear reI
clear relImg
clear testImgNames
clear testPath
clear trainImgSet
clear trainlabel
clear trainPath
clear trainSet
clear faceImgTest
clear testLabel
clear trainLabel
clear xRel


%% Create and train

net = classifiarSiameseNetwork.createNetwork();

[fcParams, net] = classifiarSiameseNetwork.trainNetwork(net, xTrain, yTrain);


%% eval
% xTest = xTrain;
% yTest = yTrain;
% accuracy = zeros(1,10);
% accuracyBatchSize = 150;
% executionEnvironment = "auto";
% for i = 1:10
%     [X1,X2,pairLabelsAcc] = classifiarSiameseNetwork.getSiameseBatch(xTest,yTest,accuracyBatchSize);
% 
%     X1 = dlarray(X1,"SSCB");
%     X2 = dlarray(X2,"SSCB");
% 
%     % If using a GPU, then convert data to gpuArray.
%     if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%         X1 = gpuArray(X1);
%         X2 = gpuArray(X2);
%     end
% 
%     % Evaluate predictions using trained network
%     Y = classifiarSiameseNetwork.predictSiamese(net,fcParams,X1,X2);
% 
%     % Convert predictions to binary 0 or 1
%     Y = gather(extractdata(Y));
%     Y = round(Y);
% 
%     % Compute average accuracy for the minibatch
%     accuracy(i) = sum(Y == pairLabelsAcc)/accuracyBatchSize
% end
% averageAccuracy = mean(accuracy)*100
 

%% eval
xTest = xTrain;
yTest = yTrain;
accuracy = zeros(1,10);
accuracyBatchSize = 150;
executionEnvironment = "auto";
for i = 1:10
    [X1,X2,pairLabelsAcc] = classifiarSiameseNetwork.getSiameseBatch(xTest,yTest,accuracyBatchSize);

    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

    % If using a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
    end

    % Evaluate predictions using trained network
    F1 = forward(net,X1);
    F2 = forward(net,X2);
    delta = 1e-6;
    Y = sqrt(sum((F1 - F2).^2,1) + delta);

    % Convert predictions to binary 0 or 1
    Y = gather(extractdata(Y));
    Y = round(Y);

    % Compute average accuracy for the minibatch
    accuracy(i) = sum(Y == pairLabelsAcc)/accuracyBatchSize
end
averageAccuracy = mean(accuracy)*100
