clear all;
close all;
clc;
trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here
%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images

%% option
option.rotate = 0;
option.reflection = 0;
option.blur = 0;
option.color = 0;
option.lightOn = 0;
option.lightOff = 0;

%% some operations to extend dataset

detectedImgs = faceDetection(trainImgSet, []);

nnInputSize = [size(detectedImgs,1),size(detectedImgs,2)];

% data augmentation
nExtend = option.rotate + option.reflection + option.blur + ...
    option.color + option.lightOn + option.lightOff;
nLabel = size(trainPersonID, 1);
faceImgs = zeros(nnInputSize(1),nnInputSize(2),...
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
            xRel = randomAffine2d(Rotation = [-30 30]);
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
            faceImgs(:,:,:,(nExtend+1)*i-j) = newI(:,:,:,j+1);
        end
    end
else
    for i = 1 : size(detectedImgs,4)
        currImg = detectedImgs(:,:,:,i);
        faceImgs(:,:,:,i) = imresize(currImg, nnInputSize);
    end
end


trainLabel = [];
for i = 1 : nExtend+1 
    trainLabel = [trainLabel;(1:nLabel)];
end
trainLabel = trainLabel(:);


%% related parameters
fprintf("Feature extraction ==== start ==== \n\n");

IMAGE_RESIZE = [[128 128]; [100 100]; [64 64]; [150 150]];
BLOCK_SIZE = [1,3,5,7,9];

% lbp
lbp_resize = IMAGE_RESIZE(2,:);
lbp_blocksize = BLOCK_SIZE(2);

trainFeatureSet = [];
meanFace = [];
eigenFaces = [];
SVMmodel = {};

%% load test img
testImgNames=dir([testPath,'*.jpg']);
faceImgTest = zeros(size(faceImgs,1),size(faceImgs,2),3,size(testImgNames,1));
for i = 1:size(testImgNames,1)
    img = imread([testPath, testImgNames(i,:).name]);%load one of the test images
    faceImgTest(:,:,:,i) = faceDetection(img, [size(faceImgs,1) size(faceImgs,2)]);
end

%% test hp
for n = 2
    tic;
    lbp_blocksize = BLOCK_SIZE(n);
    fprintf("Feature extraction ==== start ====\n\n");
    % feature extraction
    [meanFace, eigenFaces, trainFeatureSet] = featureLBP.extractFeatureSet(faceImgs,lbp_resize,lbp_blocksize);

    % To use svm to predict, we need to pre-train a svm model
    SVMmodel = fitcecoc(trainFeatureSet', trainLabel, 'Coding', 'onevsall');

    fprintf("Feature extraction ==== Finish ====\n\n");
    % face Recognition
%%
    outputID=strings([size(testImgNames,1),1]);
    for i = 1:size(testImgNames,1)
        featureTest = featureLBP.extractFeature(meanFace, eigenFaces,...
            faceImgTest(:,:,:,i), lbp_resize, lbp_blocksize);

        index = predict(SVMmodel, featureTest');
        outputID(i,:) = trainPersonID(index,:);

        fprintf("sample : %d  predict as object: %s\n", i, outputID(i,:));
    end
%%
    runTime=toc;

    % TEST

    load testLabel
    correctP=0;
    for i=1:size(testLabel,1)
        if strcmp(outputID(i,:),testLabel(i,:))
            correctP=correctP+1;
        end
    end
    recAccuracy=correctP/size(testLabel,1)*100;  %Recognition accuracy
    
    fprintf("block size: %d, acc: %d, run time: %d\n", lbp_blocksize, recAccuracy, runTime);
end

