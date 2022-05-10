clear;
clc;
%% Retrive training and testing images

trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here


%% option
option.rotate = 0;
option.reflection = 1;
option.blur = 0;
option.color = 1;
option.lightOn = 1;
option.lightOff = 1;

        
%% Load and Preprocess --- train data

nnInputSize = [100 100];
[trainImgSet, trainPersonID]=loadTrainingSet(trainPath);

detectedImgs = faceDetection(trainImgSet, []);
inputW = size(trainImgSet, 2);
inputH = size(trainImgSet, 1);

% data augmentation
nExtend = option.rotate + option.reflection + option.blur + ...
    option.color + option.lightOn + option.lightOff;
nLabel = size(trainPersonID, 1);
trainSet = zeros(2048,size(detectedImgs,4)*(nExtend+1), 'uint8');
if nExtend > 0
    for i = 1 : size(detectedImgs,4)
        newI = zeros([nnInputSize 3 (nExtend+1)], 'uint8');
        index = 1;
        currImg = detectedImgs(:,:,:,i);
        newI(:,:,:,index) = imresize(currImg, nnInputSize);
        
        index = index + 1;
        
        if option.rotate == 1
            rol = randomAffine2d("Rotation", [-30 30]);
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
            singleI = single(newI(:,:,:,j+1));
            dlI = dlarray(singleI,"SSCB");
            pred = predict(net,dlI,"Output","fc15");
            trainSet(:,(nExtend+1)*i-j) = extractdata(pred);
        end
    end
else
    for i = 1 : size(detectedImgs,4)
        currImg = detectedImgs(:,:,:,i);
        img = imresize(currImg, nnInputSize);
        singleI = single(img);
        dlI = dlarray(singleI, "SSCB");
        trainSet(:,i) = extractdata(predict(net,dlI));
    end
end


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
xTest = [];
for i = 1:size(testImgNames,1)
    faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
    faceImgTest = faceDetection(faceImgTest, nnInputSize);
    singleI = single(faceImgTest);
    dlI = dlarray(singleI, "SSCB");
    xTest(:,i) = extractdata(predict(net, dlI,"Output","fc15"));
end

%% PCA
% [d, s, latent] = pca(trainSet', "Economy", false);
% per = cumsum(latent)./sum(latent);
% newD = length(find(per <= 1.0));
% eigen = d(:,1:newD)';
% PCAtrainSet = eigen * trainSet;
PCAtrainSet = trainSet;

%% train
SVMmodel = fitcecoc(PCAtrainSet', trainLabel, 'Coding', 'onevsall');


%% test

for i = 1:size(testImgNames,1)
    % test = eigen * xTest(:,i);
    test = xTest(:,i);
    
    index = predict(SVMmodel, test');
 	outputID(i,:) = trainPersonID(index(1),:);
    fprintf("%d, predict: %s \n", i, outputID(i,:));
end

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