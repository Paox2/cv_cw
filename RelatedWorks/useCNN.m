clear;
clc;
%% Retrive training and testing images
%load own_93_7210_70.mat
load own_90_5460_54.mat
%load own_90_3019_30.mat

trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here


%% option
option.rotateLeft = 0;
option.rotateRight = 0;
option.reflection = 1;
option.blur = 1;
option.color = 0;
option.lightOn = 1;
option.lightOff = 1;


%% Load and Preprocess --- train data

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath);
tic;
detectedImgs = faceDetection(trainImgSet, []);
inputW = size(detectedImgs, 2);
inputH = size(detectedImgs, 1);

% data augmentation
nExtend = option.rotateLeft + option.rotateRight + option.reflection +...
    option.blur + option.color + option.lightOn + option.lightOff;
nLabel = size(trainPersonID, 1);
inputSize = net.Layers(1).InputSize;
imdsTrain = zeros(inputW,inputH,3,size(detectedImgs,4)*(nExtend+1), 'uint8');
if nExtend > 0
    for i = 1 : size(detectedImgs,4)
        newI = zeros([inputW,inputH 3 (nExtend+1)], 'uint8');
        index = 1;
        currImg = detectedImgs(:,:,:,i);
        newI(:,:,:,index) = currImg;
        
        index = index + 1;
        
        if option.rotateLeft == 1
            rotImg = imrotate(currImg, 20);
            newI(:,:,:,index) = imresize(rotImg, [inputW,inputH]);
            index = index + 1;
        end
        
        if option.rotateRight== 1
            rotImg = imrotate(currImg, -20);
            newI(:,:,:,index) = imresize(rotImg, [inputW,inputH]);
            index = index + 1;
        end
        
        if option.reflection == 1
            for k=1:3
                relImg(:,:,k)=fliplr(currImg(:,:,k));
            end
            newI(:,:,:,index) = relImg;
            index = index + 1;

        end
        
        if option.lightOn == 1
            reI = currImg;
            reI(reI > 225) = 225;
            lightI = reI + 30;
            newI(:,:,:,index) = lightI;
            index = index + 1;
        end
        
        if option.lightOff == 1
            reI = currImg;
            reI(reI < 30) = 30;
            lightI = reI - 30;
            newI(:,:,:,index) = lightI;
            index = index + 1;
        end
        
        if option.color == 1
            lightImg = jitterColorHSV(currImg,'Contrast',0.4,'Hue',0.1,'Saturation',0.2,'Brightness',0.3);
            newI(:,:,:,index) = lightImg;
            index = index + 1;
        end
        
        if option.blur == 1
            noiseImg = imnoise(currImg,'gaussian');
            newI(:,:,:,index) = noiseImg;
            index = index + 1;
        end
        
        for j = 0 : (index - 2)
            singleI = single(newI(:,:,:,j+1));
            imdsTrain(:,:,:,(nExtend+1)*i-j) = singleI;
        end
    end
else
    for i = 1 : size(detectedImgs,4)
        currImg = detectedImgs(:,:,:,i);
        imdsTrain(:,:,:,i) = currImg;
    end
end

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
trainSet = activations(net,augimdsTrain,"fc15",OutputAs="columns");

trainLabel = [];
for i = 1 : (nExtend + 1)
    trainLabel = [trainLabel;(1:nLabel)];
end
trainLabel = trainLabel(:);

%% clear

clear accuracy;
clear accuracyBatchSize;
clear ans;
clear currImg;
clear detectedImgs
clear executionEnvironment
clear faceImgTest
clear hpx
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
clear option
clear opts
clear pairLabelsAcc
clear reI
clear relImg
clear faceImgTest
clear xRel
clear singleI

%% Load and Preprocess --- Test data

load testLabel;
testImgNames=dir([testPath,'*.jpg']);
outputID=strings([size(testImgNames,1),1]);
faceImgTest = imread([testPath, testImgNames(1,:).name]);
imdsTest = zeros(size(faceImgTest,1),size(faceImgTest,2),...
                3,size(testImgNames,1), 'uint8');
imdsTest(:,:,:,1) = faceImgTest;

for i = 2:size(testImgNames,1)
    faceImgTest = imread([testPath, testImgNames(i,:).name]);
    imdsTest(:,:,:,i) = single(faceImgTest);
end

detectedImgsTest = faceDetection(imdsTest, []);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),detectedImgsTest);
testSet = activations(net,augimdsTrain,"fc15",OutputAs="columns");

%% PCA
PCAtrainSet = trainSet;
%% train
SVMmodel = fitcecoc(PCAtrainSet', trainLabel, 'Coding', 'onevsall');

%% test

for i = 1:size(testImgNames,1)
    test = testSet(:,i);
    
    index = predict(SVMmodel, test');
 	outputID(i,:) = trainPersonID(index(1),:);
    fprintf("%d, predict: %s \n", i, outputID(i,:));
end
runTime=toc
%% cal result

load testLabel;

correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputID(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end

recAccuracy=correctP/size(testLabel,1)*100

conMatrix = confusionmat(testLabel, outputID);

accPerLabel = diag(conMatrix) ./ sum(conMatrix,2);