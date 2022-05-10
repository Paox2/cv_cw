classdef featurePCA
    methods(Static)

        function [meanFace, eigenFaces, trainFeatureSet] = extractFeatureSet(faceImgs, reSize)
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

                trainFeatureSet = eigenFaces * centeredImgSet;
            else
                %-------------------- number of image size ---------------
                
                trainSet = zeros(oh*ow,size(faceImgs,4));
                for i=1:size(faceImgs,4)
                    trainSet(:,i) = Preprocess(faceImgs(:,:,:,i));        
                end
            
                % cause each objects just have one image, so we cannot down
                % sample the number of images
                meanFace = mean(trainSet, 2);
                nTrainImgs = size(trainSet, 2);
                centeredImgSet = trainSet - repmat(meanFace, [1,nTrainImgs]);
                clear trainSet;
                
                % V : eigenvector matrix, D : eigenvalue matrix
                L = cov(centeredImgSet);
                [V,D] = eig(L);
                clear L;

                % sort by most dominent eigen vectors
                eigVal = diag(D);
                [eigVal, indices] = sort(eigVal,'descend');
                eigVec = V(:,indices);
                clear V;

                sizeD = floor(size(eigVec,2) * 0.9);
                downEigVec = eigVec(:,1:sizeD);
                clear eigVec;

                % norm
                normEigVec=sqrt(sum(downEigVec.^2));
                downEigVec=downEigVec./repmat(normEigVec,size(downEigVec,1),1);

                % get eigen faces
                eigenFaces = centeredImgSet * downEigVec;
                eigenFaces = eigenFaces';
                size(eigenFaces)
                clear downEigVec;

                % obtain train feature set 
                trainFeatureSet = zeros(size(eigenFaces,1),nTrainImgs);
                for i = 1 : nTrainImgs
                    temp = eigenFaces * centeredImgSet(:,i);
                    trainFeatureSet(:,i) = temp;
                end
                clear centeredImgSet;
            end
           
            
%             for i = 1:10
%                 img = reshape(eigenFaces(:,i)', [m,n]);
%                 img = img';
%                 imshow(img);
%             end
        end
    
        
        
        function feature = extractFeature(meanFace, eigenFaces, faceImg)
            
            faceImg = faceImg - meanFace;
            feature =  eigenFaces * faceImg;
        end
        
    end
end