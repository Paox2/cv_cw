clear all;
close all;  
%% Retrive training and testing images

trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here


%% hp
hp.learning_rate = 0.0001;
hp.maxEpochs = 20;
hp.miniBatchSize = 64;


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


%% option
option.rotate = 0;
option.reflection = 1;
option.blur = 0;
option.color = 1;
option.lightOn = 1;
option.lightOff = 1;

        
%% Load and Preprocess --- train data

nnInputSize = [224 224];
[trainImgSet, trainPersonID]=loadTrainingSet(trainPath);

detectedImgs = faceDetection(trainImgSet, []);
inputW = size(trainImgSet, 2);
inputH = size(trainImgSet, 1);

% data augmentation
nExtend = option.rotate + option.reflection + option.blur + ...
    option.color + option.lightOn + option.lightOff;
nLabel = size(trainPersonID, 1);
trainSet = zeros(nnInputSize(1),nnInputSize(2),...
    size(detectedImgs,3),size(detectedImgs,4)*(nExtend+1), 'uint8');
if nExtend > 0
    for i = 1 : size(detectedImgs,4)
        newI = [];
        index = 1;
        currImg = detectedImgs(:,:,:,i);
        newI(:,:,:,index) = imresize(currImg, nnInputSize);
        
        index = index + 1;
        
        if option.rotate == 1
            rol = randomAffine2d(Rotation = [-30 30]);
            rotImg = imwarp(currImg, rol);
            newI(:,:,:,index) = imresize(rotImg, nnInputSize);
            index = index + 1;
        end
        
        if option.reflection == 1
            xRel = randomAffine2d("XReflection", true);
            relImg = imwarp(currImg, xRel);
            newI(:,:,:,index) = imresize(relImg, nnInputSize);
            index = index + 1;
        end
        
        if option.lightOn == 1
            reI = imresize(currImg, nnInputSize);
            reI(reI > 225) = 225;
            lightI = reI + 30;
            newI(:,:,:,index) = lightI;
            index = index + 1;
        end
        
        if option.lightOff == 1
            reI = imresize(currImg, nnInputSize);
            reI(reI < 30) = 30;
            lightI = reI - 30;
            newI(:,:,:,index) = lightI;
            index = index + 1;
        end
        
        if option.color == 1
            lightImg = jitterColorHSV(currImg,'Contrast',0.4,'Hue',0.1,'Saturation',0.2,'Brightness',0.3);
            newI(:,:,:,index) = imresize(lightImg, nnInputSize);
            index = index + 1;
        end
        
        if option.blur == 1
            noiseImg = imnoise(currImg,'gaussian');
            newI(:,:,:,index) = imresize(noiseImg, nnInputSize);
            index = index + 1;
        end
        
        for j = 0 : (index - 2)
            trainSet(:,:,:,(nExtend+1)*i-j) = newI(:,:,:,j+1);
        end
    end
else
    for i = 1 : size(detectedImgs,4)
        currImg = detectedImgs(:,:,:,i);
        trainSet(:,:,:,i) = imresize(currImg, nnInputSize);
    end
end


trainLabel = [];
for i = 1 : nExtend+1 
    trainLabel = [trainLabel;(1:nLabel)];
end
trainLabel = trainLabel(:);


% get train and label set
yTrain = categorical(trainLabel);
xTrain = trainSet;


%% Load and Preprocess --- Test data

load testLabel;
testImgNames=dir([testPath,'*.jpg']);
xTest = zeros(nnInputSize(1),nnInputSize(2),3,size(testImgNames,1));
for i = 1:size(testImgNames,1)
    faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
    faceImgTest = faceDetection(faceImgTest, nnInputSize);
    xTest(:,:,:,i) = double(faceImgTest);   
end
yTest = categorical(cellstr(testLabel));
clear testLabel;

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
%clear trainPersonID
clear trainSet
clear faceImgTest
%clear xTest
%clear yTest
clear testLabel
clear trainLabel
clear xRel


%% Create and train

net = classifiarSiameseNetwork.createNetwork();

[fcParams, net] = classifiarSiameseNetwork.trainNetwork(net, xTrain, yTrain);


%% eval

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
    Y = classifiarSiameseNetwork.predictSiamese(net,fcParams,X1,X2);

    % Convert predictions to binary 0 or 1
    Y = gather(extractdata(Y));
    Y = round(Y);

    % Compute average accuracy for the minibatch
    accuracy(i) = sum(Y == pairLabelsAcc)/accuracyBatchSize
end
averageAccuracy = mean(accuracy)*100

% if averageAccuracy > 90
%     save net;
% end   