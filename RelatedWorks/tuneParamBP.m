clear all;
close all;  
trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here
%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images



%% hp setting
%HIDEN = [80,100,120,200];
HIDEN = [40];
BPparameterFin = ['tansig'];
BPparameterFout = ['logsig'];
% BPparameterFtrain = ['trainscg';'traingdx'];
BPparameterFtrain = ['traingdx'];
EPOCHS = 20000;
%GOAL = [0,0.00005,0.0001,0.0003];
GOAL = [0];
%LR = [0.001,0.01,0.1];
LR = 0.01;

pca_resize = [100 100];


%% Pre process
detectedImgs = faceDetection(trainImgSet, []);
nExtend = 3;
nLabel = size(trainPersonID, 1);
faceImgs = zeros(size(detectedImgs,1),size(detectedImgs,2),...
    size(detectedImgs,3),size(detectedImgs,4)*(nExtend+1), 'uint8');
if nExtend > 0
    faceImgs(:,:,:,1:(nExtend+1):end) = detectedImgs;
    for i = 1 : size(detectedImgs,4)
        currImg = detectedImgs(:,:,:,i);
        xRel = randomAffine2d(Rotation = [-30 30]);
        relImg3 = imwarp(currImg, xRel);
        relImg3 = imresize(relImg3, [size(detectedImgs,1), size(detectedImgs,2)]);
        relImg1 = jitterColorHSV(currImg,'Contrast',0.4,'Hue',0.1,'Saturation',0.2,'Brightness',0.3);
        relImg2 = imnoise(currImg,'gaussian');
        faceImgs(:,:,:,(nExtend+1)*i-1) = relImg1;
        faceImgs(:,:,:,(nExtend+1)*i) = relImg2;
        faceImgs(:,:,:,(nExtend+1)*i-2) = relImg3;
    end
else
    faceImgs = detectedImgs;
end

trainLabel = [];
for i = 1 : nExtend+1 
    trainLabel = [trainLabel;(1:nLabel)];
end
trainLabel = trainLabel(:);


[meanFace, eigenFaces, trainFeatureSet] = featurePCA.extractFeatureSet(faceImgs, pca_resize);

testImgNames=dir([testPath,'*.jpg']);
outputID=strings([size(testImgNames,1),1]);
testList = [];
for i = 1:size(testImgNames,1)
    faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
    faceImgTest = faceDetection(faceImgTest, [size(faceImgs,1) size(faceImgs,2)]);

    if ~isempty(pca_resize) 
        faceImgTest = imresize(faceImgTest, pca_resize);
    end
    faceImgTest = Preprocess(faceImgTest);
    faceImgTest = featurePCA.extractFeature(meanFace, eigenFaces, faceImgTest);
    testList(:,i) = faceImgTest;
end


%% train
for a = 1
    for b = 1
            BPparameter.hideN = HIDEN(b);
            BPparameter.fIn = BPparameterFin(1,:);
            BPparameter.fOut = BPparameterFout(1,:);
            BPparameter.fTrain = BPparameterFtrain(1,:);
            BPparameter.epochs = EPOCHS;
            BPparameter.goal = GOAL(a);
            BPparameter.lr = LR;
            BPparameter.divideFcn = "";

            tic;

            net = classifiarBP.BPtrain(trainFeatureSet, BPparameter, trainLabel);

            %fprintf("classify start\n\n");

            for i = 1:size(testImgNames,1)
                outputID(i) = classifiarBP.BPtest(trainPersonID, testList(:,i), net);
                %fprintf("sample : %d  predict as object: %s\n", i, outputID(i,:));
            end

            runTime=toc;

            load testLabel
            correctP=0;
            for i=1:size(testLabel,1)
                if strcmp(outputID(i,:),testLabel(i,:))
                    correctP=correctP+1;
                end
            end
            recAccuracy=correctP/size(testLabel,1)*100;
            
            fprintf("Goal: %f ; LR: %f ; hidenN: %d ;  time: %f ; acc: %f\n",...
                GOAL(a), 0.01, HIDEN(b), runTime, recAccuracy);
    end
end


