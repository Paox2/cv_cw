
clear all;
close all;  

load testLabel;
trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here
%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images

faceImgs = faceDetection(trainImgSet, []);

KERNEL_TYPE = ['poly','fpp','tanh','gaussian','simple'];



testImgNames=dir([testPath,'*.jpg']);
outputID=strings([size(testImgNames,1),1]);

%% 

kernelType = 'poly';
kernelArgs = [1 5];
kpcaModel = {};
kpcaModel.kernelArgs = kernelArgs;
kpcaModel.kernelType = kernelType;
for recogMethod = 2
    for poly = 2 : 4
        kpcaModel.kernelArgs(2) = poly;
        [kpcaModel, trainFeatureSet] = featureKPCA.extractFeatureSet(faceImgs, kpcaModel);

        for i = 1:size(testImgNames,1)
            faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
            faceImgTest = faceDetection(faceImgTest, [size(faceImgs,1) size(faceImgs,2)]);

            faceImgTest = Preprocess(faceImgTest);
            faceImgTest = featureKPCA.extractFeature(kpcaModel, faceImgTest);

            if recogMethod == 2
                SVMmodel = fitcecoc(trainFeatureSet', trainPersonID, ...
                    'Coding', 'onevsall');
            end

            if recogMethod == 0
                outputID(i) = recogCV(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 1
                outputID(i) = recogEM(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 2
                outputID(i) = recogSVM(faceImgTest, SVMmodel);
            elseif recogMethod == 3
                outputID(i) = recogKNN(trainPersonID, faceImgTest, trainFeatureSet);
            end
        end

        correctP=0;
        for i=1:size(testLabel,1)
            if strcmp(outputID(i,:),testLabel(i,:))
                correctP=correctP+1;
            end
        end
        recAccuracy = correctP/size(testLabel,1)*100;  %Recognition accuracy
        fprintf("poly:  %d  recog:  %d  recAccuracy: %f\n ", poly, recogMethod, recAccuracy);
    end
end


%% 

kernelType = 'fpp';
kernelArgs = [1 5];
kpcaModel = {};
kpcaModel.kernelArgs = kernelArgs;
kpcaModel.kernelType = kernelType;

for recogMethod = 2
    for fpp = 0.6:0.1:0.9
        kpcaModel.kernelArgs(2) = fpp;
        [kpcaModel, trainFeatureSet] = featureKPCA.extractFeatureSet(faceImgs, kpcaModel);

        for i = 1:size(testImgNames,1)
            faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
            faceImgTest = faceDetection(faceImgTest, [size(faceImgs,1) size(faceImgs,2)]);

            faceImgTest = Preprocess(faceImgTest);
            faceImgTest = featureKPCA.extractFeature(kpcaModel, faceImgTest);

            if recogMethod == 2
                SVMmodel = fitcecoc(trainFeatureSet', trainPersonID, ...
                    'Coding', 'onevsall');
            end

            if recogMethod == 0
                outputID(i) = recogCV(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 1
                outputID(i) = recogEM(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 2
                outputID(i) = recogSVM(faceImgTest, SVMmodel);
            elseif recogMethod == 3
                outputID(i) = recogKNN(trainPersonID, faceImgTest, trainFeatureSet);
            end
        end

        correctP=0;
        for i=1:size(testLabel,1)
            if strcmp(outputID(i,:),testLabel(i,:))
                correctP=correctP+1;
            end
        end
        recAccuracy = correctP/size(testLabel,1)*100;  %Recognition accuracy
        
        fprintf("fpp:  %f  recog:  %d  recAccuracy: %f\n ", fpp, recogMethod, recAccuracy);
    end
end


%% 

kernelType = 'simple';
kernelArgs = [];
kpcaModel = {};
kpcaModel.kernelArgs = kernelArgs;
kpcaModel.kernelType = kernelType;


for recogMethod = 2
        [kpcaModel, trainFeatureSet] = featureKPCA.extractFeatureSet(faceImgs, kpcaModel);

        for i = 1:size(testImgNames,1)
            faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
            faceImgTest = faceDetection(faceImgTest, [size(faceImgs,1) size(faceImgs,2)]);

            faceImgTest = Preprocess(faceImgTest);
            faceImgTest = featureKPCA.extractFeature(kpcaModel, faceImgTest);

            if recogMethod == 2
                SVMmodel = fitcecoc(trainFeatureSet', trainPersonID, ...
                    'Coding', 'onevsall');
            end

            if recogMethod == 0
                outputID(i) = recogCV(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 1
                outputID(i) = recogEM(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 2
                outputID(i) = recogSVM(faceImgTest, SVMmodel);
            elseif recogMethod == 3
                outputID(i) = recogKNN(trainPersonID, faceImgTest, trainFeatureSet);
            end
        end

        correctP=0;
        for i=1:size(testLabel,1)
            if strcmp(outputID(i,:),testLabel(i,:))
                correctP=correctP+1;
            end
        end
        recAccuracy = correctP/size(testLabel,1)*100;  %Recognition accuracy
        fprintf("simple  recog:  %d  recAccuracy: %f\n ", recogMethod, recAccuracy);

end

%% 

kernelType = 'tanh';
kernelArgs = 0;
kpcaModel = {};
kpcaModel.kernelArgs = kernelArgs;
kpcaModel.kernelType = kernelType;

for recogMethod = 2
        [kpcaModel, trainFeatureSet] = featureKPCA.extractFeatureSet(faceImgs, kpcaModel);

        for i = 1:size(testImgNames,1)
            faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
            faceImgTest = faceDetection(faceImgTest, [size(faceImgs,1) size(faceImgs,2)]);

            faceImgTest = Preprocess(faceImgTest);
            faceImgTest = featureKPCA.extractFeature(kpcaModel, faceImgTest);

            if recogMethod == 2
                SVMmodel = fitcecoc(trainFeatureSet', trainPersonID, ...
                    'Coding', 'onevsall');
            end

            if recogMethod == 0
                outputID(i) = recogCV(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 1
                outputID(i) = recogEM(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 2
                outputID(i) = recogSVM(faceImgTest, SVMmodel);
            elseif recogMethod == 3
                outputID(i) = recogKNN(trainPersonID, faceImgTest, trainFeatureSet);
            end
        end

        correctP=0;
        for i=1:size(testLabel,1)
            if strcmp(outputID(i,:),testLabel(i,:))
                correctP=correctP+1;
            end
        end
        recAccuracy = correctP/size(testLabel,1)*100;  %Recognition accuracy
        fprintf("tanh  recog:  %d  recAccuracy: %f\n ", recogMethod, recAccuracy);
end

%% 

kernelType = 'gaussian';
kernelArgs = 0;
kpcaModel = {};
kpcaModel.kernelArgs = kernelArgs;
kpcaModel.kernelType = kernelType;


for recogMethod = 2
        [kpcaModel, trainFeatureSet] = featureKPCA.extractFeatureSet(faceImgs, kpcaModel);

        for i = 1:size(testImgNames,1)
            faceImgTest = imread([testPath, testImgNames(i,:).name]);%load one of the test images
            faceImgTest = faceDetection(faceImgTest, [size(faceImgs,1) size(faceImgs,2)]);

            faceImgTest = Preprocess(faceImgTest);
            faceImgTest = featureKPCA.extractFeature(kpcaModel, faceImgTest);

            if recogMethod == 2
                SVMmodel = fitcecoc(trainFeatureSet', trainPersonID, ...
                    'Coding', 'onevsall');
            end

            if recogMethod == 0
                outputID(i) = recogCV(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 1
                outputID(i) = recogEM(trainPersonID, faceImgTest, trainFeatureSet);
            elseif recogMethod == 2
                outputID(i) = recogSVM(faceImgTest, SVMmodel);
            elseif recogMethod == 3
                outputID(i) = recogKNN(trainPersonID, faceImgTest, trainFeatureSet);
            end
        end

        correctP=0;
        for i=1:size(testLabel,1)
            if strcmp(outputID(i,:),testLabel(i,:))
                correctP=correctP+1;
            end
        end
        recAccuracy = correctP/size(testLabel,1)*100;  %Recognition accuracy
        fprintf("gaussian  recog:  %d  recAccuracy: %f\n ", recogMethod, recAccuracy);

end

function resultID = recogCV(trainPersonID, faceImgTest, trainFeatureSet)
similarityValues = trainFeatureSet'*faceImgTest;
currTestId = find(similarityValues == max(similarityValues));
resultID = trainPersonID(currTestId(1),:);
end


function resultID = recogEM(trainPersonID, faceImgTest, trainFeatureSet)
Eucs = []; 
for i = 1:size(trainPersonID, 1)
    currFace = trainFeatureSet(:,i);
    dis = (norm(faceImgTest - currFace))^2;
    Eucs = [Eucs dis];
end
[~,currTestId] = min(Eucs);
resultID = trainPersonID(currTestId(1),:);
end


function resultID = recogSVM(faceImgTest, SVMmodel)
resultID = predict(SVMmodel, faceImgTest');
end 


function resultID = recogKNN(trainPersonID, faceImgTest, trainFeatureSet)
index = knnsearch(trainFeatureSet', faceImgTest', ...
    'k', size(trainPersonID,1), ...
    'Distance', 'cityblock');
resultID = trainPersonID(index(1),:);
end
