classdef feature2DPCA
    methods(Static)
    
    
        function [meanFace, eigenFaces, trainFeatureSet] = extractFeatureSet(faceImgs, reSize, newD)
            nTrainImgs = size(faceImgs, 4);
            
            greyImgSet = zeros(reSize(1),reSize(2),nTrainImgs);
            for i = 1 : nTrainImgs
                img = faceImgs(:,:,:,i);
                img = imresize(img, reSize);  
                img = Preprocess(img);
                greyImgSet(:,:,i) = reshape(img, reSize);
            end
            
            % Computing image covariance (scatter) matrix
            meanFace = mean(greyImgSet, 3);
            covMatrix = zeros(reSize(2),reSize(2));
            for i = 1 : nTrainImgs
                centeredImg = greyImgSet(:,:,i) - meanFace;
                covMatrix = covMatrix +  centeredImg' * centeredImg;
            end
            covMatrix = covMatrix / nTrainImgs;
            
            % eigen decomposition
            [eigVec, eigVal] = eig(covMatrix);
            
            eigVal = abs(diag(eigVal)');
            [~,indices] = sort(eigVal);
            eigVec = fliplr(eigVec(:,indices));
            
            eigenFaces = eigVec(:,1:newD);
            
            % derive new train feature set
            trainFeatureSet = zeros(reSize(1)*newD, nTrainImgs);
            for i = 1 : nTrainImgs
                imgF = greyImgSet(:,:,i) * eigenFaces;
                trainFeatureSet(:,i) = imgF(:);
            end
        end
        
        
        function feature = extractFeature(meanFace, eigenFaces, faceImg)
        
            faceImg = faceImg - meanFace;
            feature =  faceImg * eigenFaces;
            feature = feature(:);
        end
        
    end
end