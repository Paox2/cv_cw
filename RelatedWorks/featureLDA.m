classdef featureLDA
    methods(Static)
        
        function [meanFace, eigenFaces, vFisher, trainFeatureSet] = extractFeatureSet(faceImgs, reSize, trainLabel)
            % faceImgs: featureNumber*imageNumber double matrix

            oh = size(faceImgs,1);
            ow = size(faceImgs,2);
            
            if ~isempty(reSize)
                %-------------------- pre define size ---------------
                
                trainSet = zeros(reSize(1)*reSize(2), size(faceImgs,4));
                for i=1:size(faceImgs,4)
                    tmp = faceImgs(:,:,:,i);
                    tmp = imresize(tmp, reSize);
                    trainSet(:,i) = Preprocess(tmp);       
                end
                clear tmp;

                % cause each objects just have one image, so we cannot down
                % sample the number of images
                meanFace = mean(trainSet, 2);
                nTrainImgs = size(trainSet, 2);
                centeredImgSet = trainSet - repmat(meanFace, [1,nTrainImgs]);
                clear trainSet;

                % V : eigenvector matrix, D : eigenvalue matrix
                L = cov(centeredImgSet');
                [V,~] = eig(L);
                clear L;
                V = V(:,end:-1:1)';
                eigenFaces = V(1:reSize(1),:);
                clear V;

                projectedSet = eigenFaces * centeredImgSet;
                clear centeredImgSet;
            else
                %-------------------- number of image size ---------------
                trainSet = zeros(oh*ow, size(faceImgs,4));
                for i=1:size(faceImgs,4)
                    trainSet(:,i) = Preprocess(faceImgs(:,:,:,i));        
                end

                % cause each objects just have one image, so we cannot down
                % sample the number of images
                meanFace = mean(trainSet, 2);
                nTrainImgs = size(trainSet, 2);
                centeredImgSet = trainSet - repmat(meanFace, [1,nTrainImgs]);
                clear trainSet;

                % calculate eigenFaces
                L = cov(centeredImgSet);
                [V,~] = eig(L);
                clear L;

                eigVec = V(:,nTrainImgs:-1:1);
                eigenFaces = centeredImgSet * eigVec;
                eigenFaces = eigenFaces';
                clear eigVec;
                clear V;

                % obtain feature set into eigen space
                projectedSet = zeros(size(eigenFaces,1),nTrainImgs);
                for i = 1 : nTrainImgs
                    temp = eigenFaces * centeredImgSet(:,i);
                    projectedSet(:,i) = temp;
                end
                clear centeredImgSet;
            end
            
            % calculate within and between scatter matrix
            meanProjectedSet = mean(projectedSet, 2);
            
            classLabel = unique(trainLabel);
            nClass = size(classLabel,1);
            if ~isempty(reSize) 
                nFea = reSize(1);
            else
                nFea = nTrainImgs;
            end
            sw = zeros(nFea, nFea);
            sb = zeros(nFea, nFea);
            
            for i = 1 : nClass
                index = find(trainLabel==classLabel(i));
                classMean = mean(projectedSet(:, index),2);
                
                % cal between class
                tempSb = classMean - meanProjectedSet;
                sb = sb + (length(index)/nTrainImgs) * (tempSb * tempSb');
                
                % cal within class
                Xclass = projectedSet(:,index);
                tempSw = zeros(nFea,nFea);
                for j = 1 : length(index)
                    tempSw = tempSw+(Xclass(:,j)-classMean)*(Xclass(:,j)-classMean)';
                end
                sw = sw + (length(index)/nTrainImgs)*tempSw;
            end
            
            % calculate fisher discriminant basis
            [jEigenVec, V] = eig(sb, sw);
            V = diag(V);
            [~, indices] = sort(V,'descend');
            
            sizeD = floor(size(jEigenVec,2) * 1.0);
            indices = indices(1:sizeD);
            vFisher = jEigenVec(:,indices);
            clear V;
            clear jEigenVec;
            
            
            projectedImages_Fisher = [];
            for i = 1 : nTrainImgs
                projectedImages_Fisher(:,i) = vFisher' * projectedSet(:,i);
            end
            trainFeatureSet = projectedImages_Fisher;
            clear projectedImages_Fisher;
        end
        
        
    
        function feature = extractFeature(meanFace, eigenFaces, vFisher, faceImg)
            % faceImgs: height*width*3 unit8 matrix
 
            faceImg = faceImg - meanFace;
            feature = vFisher' * (eigenFaces * faceImg);
        end
        
    end
end