clear all;
close all;  
trainPath='.\FaceDatabase\Train\'; % These training/testing folders need to be in the same root folder of this code. 
testPath='.\FaceDatabase\Test\';   % Or you can use the full folder path here
%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images

size(trainImgSet)  % if successfully loaded this should be with dimension of 600,600,3,100

%% Now we need to do facial recognition: Baseline Method

tic;
    % featureExtractionMethod
    % 0 - use all features
    % 1 - hog
    % 2 - pca
    % 3 - pca + lda
    % 4 - pca + lbp
    % 5 - 2dpca
    % 6 - kpca
    %
    % size of trainFeatureSet N * D
    %
    % recogMethod
    % 0 - cross correlation
    % 1 - euclidean metric
    % 2 - svm
    % 3 - knn(cityblock)
    % 4 - LEM
    % 5 - Baysian
    outputID = baseLineModel(trainImgSet, trainPersonID, testPath,... 
        1, 2, 0 );
runTime=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputID(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy


