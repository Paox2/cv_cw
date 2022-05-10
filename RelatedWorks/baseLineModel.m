function outputID = baseLineModel(trainImgSet, trainPersonID, testPath, featureExtractionMethod, recogMethod, detectionOn)
% featureExtractionMethod
% 0 - use all features
% 1 - hog
% 2 - pca
% 3 - pca + lda
% 4 - lbp
% 5 - 2dpca
% 6 - kpca
%
% size of trainFeatureSet N * D
%
% recogMethod
% 0 - cross validation
% 1 - euclidean metric
% 2 - svm
% 3 - knn(cityblock)
% 4 - LEM
% 5 - Baysian


if nargin < 5
    detectionOn = 0;
end


%% option
option.rotateLeft = 0;
option.rotateRight = 0;
option.reflection = 0;
option.blur = 0;
option.color = 0;
option.lightOn = 0;
option.lightOff = 0;


%% some operations to extend dataset

if detectionOn == 1
    detectedImgs = faceDetection(trainImgSet, []);
else
    detectedImgs = trainImgSet;
end

nnInputSize = [size(detectedImgs,1),size(detectedImgs,2)];

% data augmentation
nExtend = option.rotateLeft + option.rotateRight + option.reflection +...
    option.blur + ...
    option.color + option.lightOn + option.lightOff;
nLabel = size(trainPersonID, 1);
faceImgs = zeros(nnInputSize(1),nnInputSize(2),...
    size(detectedImgs,3),size(detectedImgs,4)*(nExtend+1), 'uint8');
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
fprintf("Feature extraction ==== start ==== method: %d\n\n", featureExtractionMethod);

IMAGE_RESIZE = [[128 128]; [100 100]; [64 64]; [150 150]];

% hog
CELL_SIZE = [[8 8];[16 16];[32 32];[64 64]];
cellSize = CELL_SIZE(3,:);
% pca
pca_resize = IMAGE_RESIZE(2,:);
% lda
lda_resize = IMAGE_RESIZE(2,:);
% lbp
lbp_resize = IMAGE_RESIZE(3,:);
% 2dpca
twodpca_resize = IMAGE_RESIZE(2,:);
twodpca_newD = 100;
% kpca
KERNEL_TYPE = ['poly','fpp','tanh','gaussian','simple'];
kernelType = 'fpp';
kernelArgs = [];

if strcmp(kernelType,'poly')==1
    kernelArgs = [1 2];
elseif strcmp(kernelType,'fpp')==1
    kernelArgs = [1 .9];
elseif strcmp(kernelType,'tanh')==1
    kernelArgs = 0;
elseif strcmp(kernelType,'gaussian')==1
end
kpcaModel = {};
kpcaModel.resize = [100 100];
kpcaModel.kernelArgs = kernelArgs;
kpcaModel.kernelType = kernelType;

trainFeatureSet = [];
meanFace = [];
eigenFaces = [];
vFisher = [];
SVMmodel = {};

% LEM classify
LEM_owt = [];
LEM_w = [];
LEM_b = [];
LEM_L = 0;
LEM_m = size(faceImgs,4);

% Bayes classify
BayesModel = {};


%% feature extraction
if featureExtractionMethod == 0
    trainFeatureSet = zeros(size(faceImgs,1)*size(faceImgs,2),size(faceImgs,4));
    for i=1:size(trainImgSet,4)
        trainFeatureSet(:,i) = Preprocess(faceImgs(:,:,:,i));        
    end
elseif featureExtractionMethod == 1
    trainFeatureSet = featureHOG.extractFeatureSet(faceImgs, cellSize);
elseif featureExtractionMethod == 2
    [meanFace, eigenFaces, trainFeatureSet] = featurePCA.extractFeatureSet(faceImgs, pca_resize);
elseif featureExtractionMethod == 3
    [meanFace, eigenFaces, vFisher, trainFeatureSet] = featureLDA.extractFeatureSet(faceImgs, lda_resize, trainLabel);
elseif featureExtractionMethod == 4
    [meanFace, eigenFaces, trainFeatureSet] = featureLBP.extractFeatureSet(faceImgs, lbp_resize, 1);
elseif featureExtractionMethod == 5
    [meanFace, eigenFaces, trainFeatureSet] = feature2DPCA.extractFeatureSet(faceImgs, twodpca_resize, twodpca_newD);
elseif featureExtractionMethod == 6
    [kpcaModel, trainFeatureSet] = featureKPCA.extractFeatureSet(faceImgs, kpcaModel);
end

% To use svm to predict, we need to pre-train a svm model
if recogMethod == 2
    SVMmodel = fitcecoc(trainFeatureSet', trainLabel, 'Coding', 'onevsall');
elseif recogMethod == 4
    LEM_L = size(trainFeatureSet,1);
    [LEM_owt,LEM_w,LEM_b] = classifiarLEM.LEMtrain(trainFeatureSet, LEM_L, LEM_m);
elseif recogMethod == 5
    %%% need more sample, can combine with svm pertu
    BayesModel = fitcnb(trainFeatureSet', trainLabel');
end


fprintf("Feature extraction ==== Finish ====\n\n");


%%  face Recognition
testImgNames=dir([testPath,'*.jpg']);
outputID=strings([size(testImgNames,1),1]);

fprintf("Face recognition ==== start ==== method: %d\n\n", recogMethod);


for i = 1:size(testImgNames,1)
    faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
    if detectionOn == 1
        faceImgTest = faceDetection(faceImgTest, [size(faceImgs,1) size(faceImgs,2)]);
    end
    
    
    if featureExtractionMethod == 0
        faceImgTest = Preprocess(faceImgTest);
    elseif featureExtractionMethod == 1
        faceImgTest = featureHOG.extractFeature(faceImgTest, cellSize);
    elseif featureExtractionMethod == 2
        if ~isempty(pca_resize) 
            faceImgTest = imresize(faceImgTest, pca_resize);
        end
            faceImgTest = Preprocess(faceImgTest);
            faceImgTest = featurePCA.extractFeature(meanFace, eigenFaces, faceImgTest);
    elseif featureExtractionMethod == 3
        if ~isempty(lda_resize) 
            faceImgTest = imresize(faceImgTest, lda_resize);
        end
        faceImgTest = Preprocess(faceImgTest);
        faceImgTest = featureLDA.extractFeature(meanFace, eigenFaces, vFisher, faceImgTest);
    elseif featureExtractionMethod == 4
        faceImgTest = featureLBP.extractFeature(meanFace, eigenFaces, faceImgTest, lbp_resize, 1);
    elseif featureExtractionMethod == 5
        if ~isempty(twodpca_resize) 
            faceImgTest = imresize(faceImgTest, twodpca_resize);
        end
        faceImgTest = Preprocess(faceImgTest);
        faceImgTest = reshape(faceImgTest, twodpca_resize);
        faceImgTest = feature2DPCA.extractFeature(meanFace, eigenFaces, faceImgTest);
    elseif featureExtractionMethod == 6
        faceImgTest = Preprocess(faceImgTest);
        faceImgTest = featureKPCA.extractFeature(kpcaModel, faceImgTest);
    end

    if recogMethod == 0
        outputID(i) = recogCV(trainLabel, faceImgTest, trainFeatureSet, trainPersonID);
    elseif recogMethod == 1
        outputID(i) = recogEM(trainLabel, faceImgTest, trainFeatureSet, trainPersonID);
    elseif recogMethod == 2
        outputID(i) = recogSVM(faceImgTest, SVMmodel, trainPersonID);
    elseif recogMethod == 3
        outputID(i) = recogKNN(trainLabel, faceImgTest, trainFeatureSet, trainPersonID);
    elseif recogMethod == 4
        % because this method perform worst, so we do not consider the data
        % augementation for this classifiar (not update any more)
        outputID(i) = classifiarLEM.LEMtest(trainLabel, faceImgTest,...
            LEM_owt,LEM_w,LEM_b, trainPersonID);
    elseif recogMethod == 5
        outputID(i) = recogBayes(faceImgTest, BayesModel, trainPersonID);
    end
    
    fprintf("sample : %d  predict as object: %s\n", i, outputID(i,:));
end

fprintf("Face recognition ==== Finish ====\n\n");

end


function resultID = recogCV(trainLabel, faceImgTest, trainFeatureSet, trainPersonID)
similarityValues = trainFeatureSet'*faceImgTest;
currTestId = find(similarityValues == max(similarityValues));
index = trainLabel(currTestId(1));
resultID = trainPersonID(index,:);
end


function resultID = recogEM(trainLabel, faceImgTest, trainFeatureSet, trainPersonID)
Eucs = []; 
for i = 1:size(trainLabel, 1)
    currFace = trainFeatureSet(:,i);
    dis = (norm(faceImgTest - currFace))^2;
    Eucs = [Eucs dis];
end
[~,currTestId] = min(Eucs);
index = trainLabel(currTestId(1));
resultID = trainPersonID(index,:);
end


function resultID = recogSVM(faceImgTest, SVMmodel, trainPersonID)
index = predict(SVMmodel, faceImgTest');
resultID = trainPersonID(index,:);
end 


function resultID = recogKNN(trainLabel, faceImgTest, trainFeatureSet, trainPersonID)
index = knnsearch(trainFeatureSet', faceImgTest', ...
    'k', 1, ...
    'Distance', 'cityblock');
i = trainLabel(mode(index));
resultID = trainPersonID(i,:);
end

function resultID = recogBayes(faceImgTest, BayesModel, trainPersonID)
index = predict(BayesModel, faceImgTest');
resultID = trainPersonID(index,:);
end

