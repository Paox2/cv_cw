classdef featureHOG
    methods(Static)
        
        function trainFeatureSet = extractFeatureSet(faceImgs, cellSize)
            % faceImgs: featureNumber*imageNumber double matrix
            img = faceImgs(:,:,:,1);
            [hog, ~] = extractHOGFeatures(img,'CellSize',cellSize);
            hogFeatureSize = length(hog)
            trainFeatureSet = zeros(hogFeatureSize, size(faceImgs,4), 'single');

            for i = 1:size(faceImgs,4)
                img = faceImgs(:,:,:,i);
                img = imbinarize(rgb2gray(img));
                trainFeatureSet(:, i) = extractHOGFeatures(img, 'CellSize', cellSize)';  
            end
        end
    
        function feature = extractFeature(faceImg, cellSize)
            % faceImgs: height*width*3 unit8 matrix
            faceImg = imbinarize(rgb2gray(faceImg));
            feature = extractHOGFeatures(faceImg, 'CellSize', cellSize)';
        end
        
    end
end