function  outputID=FaceRecognitionNew(trainImgSet, trainPersonID, testPath)

%% load pre-trained net
load network.mat;

%% Face detection
detectedImgs = faceDetection(trainImgSet, []);
inputW = size(detectedImgs, 2);
inputH = size(detectedImgs, 1);


%% Data augmentation
nLabel = size(trainPersonID, 1);
inputSize = net.Layers(1).InputSize;
imdsTrain = zeros(inputW,inputH,3,size(detectedImgs,4)*5, 'uint8');
for i = 1 : size(detectedImgs,4)
    % save original imgs
    currImg = detectedImgs(:,:,:,i);
    imdsTrain(:,:,:,5*i-4) = currImg;
    
    % save flip image
    for k=1:3
        relImg(:,:,k)=fliplr(currImg(:,:,k));
    end
    imdsTrain(:,:,:,5*i-3) = relImg;

    % save brighter image
    reI = currImg;
    reI(reI > 225) = 225;
    lightI = reI + 30;
    imdsTrain(:,:,:,5*i-2) = lightI;
        
    % save brighter image
    reI = currImg;
    reI(reI < 30) = 30;
    lightI = reI - 30;
    imdsTrain(:,:,:,5*i-1) = lightI;

    % blur
    noiseImg = imnoise(currImg,'gaussian');
    imdsTrain(:,:,:,5*i) = noiseImg;
end


%% generate features list of train set and labels
% generate feature lists of training dataset
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
trainSet = activations(net,augimdsTrain,"fc15",OutputAs="columns");

% generate labels
trainLabel = [];
for i = 1 : 5
    trainLabel = [trainLabel;(1:nLabel)];
end
trainLabel = trainLabel(:);


%% generate features list of test set and labels
% load test set
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

% face detection of images in test dataset
detectedImgsTest = faceDetection(imdsTest, []);

% extract feature of test dataset
augimdsTrain = augmentedImageDatastore(inputSize(1:2),detectedImgsTest);
testSet = activations(net,augimdsTrain,"fc15",OutputAs="columns");


%% train SVM model
SVMmodel = fitcecoc(trainSet', trainLabel, 'Coding', 'onevsall');


%% recognize

for i = 1:size(testImgNames,1)
    test = testSet(:,i);
    index = predict(SVMmodel, test');
 	outputID(i,:) = trainPersonID(index(1),:);
end

end

