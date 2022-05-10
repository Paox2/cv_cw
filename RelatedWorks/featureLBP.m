classdef featureLBP
    methods(Static)

        function [meanFace, eigenFaces, trainFeatureSet] = extractFeatureSet(faceImgs, reSize, blockSize)
            % faceImgs: featureNumber*imageNumber double matrix
        
            h = size(faceImgs,1);
            w = size(faceImgs,2);
            feature = uint8(zeros(h, w));
            neighbor = uint8(zeros(8, 1));
            filterImgs = zeros(size(faceImgs,1), size(faceImgs,2), size(faceImgs,4));
            
            for i = 1 : size(faceImgs,4)
                faceImg = faceImgs(:,:,:,i);
                img = Preprocess(faceImg);
                img = reshape(img, [h, w]);

                for y = (blockSize+1) : (w - blockSize)
                    for x = (blockSize+1) : (h - blockSize)
                        center = img(x, y);
                        neighbor(8) = img(x-blockSize, y-blockSize) >= center;
                        neighbor(7) = img(x-blockSize, y  ) >= center;
                        neighbor(6) = img(x-blockSize, y+blockSize) >= center;
                        neighbor(5) = img(x  , y+blockSize) >= center;
                        neighbor(4) = img(x+blockSize, y+blockSize) >= center;
                        neighbor(3) = img(x+blockSize, y  ) >= center;
                        neighbor(2) = img(x+blockSize, y-blockSize) >= center;
                        neighbor(1) = img(x  , y-blockSize) >= center;
                        patt = 0;
                        for k = 1:8
                            patt = patt + neighbor(k) * bitshift(1, k-1);
                        end
                        feature(x, y) = patt;
                    end
                end
                filterImgs(:,:,i) = feature;
            end
            
            trainSet = zeros(reSize(1)*reSize(2), size(faceImgs,4));
            for i=1:size(filterImgs,3)
                tmp = filterImgs(:,:,i);
                tmp = imresize(tmp, reSize);  
                trainSet(:,i) = Preprocess(tmp);       
            end

            
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
        end
        
        
        
        function feature = extractFeature(meanFace, eigenFaces, faceImgTest, reSize, blockSize)
            h = size(faceImgTest,1);
            w = size(faceImgTest,2);
            feature = uint8(zeros(h, w));
            neighbor = uint8(zeros(8, 1));
            
            img = Preprocess(faceImgTest);
            img = reshape(img, [h, w]);

            for y = (blockSize+1) : (w - blockSize)
                for x = (blockSize+1) : (h - blockSize)
                    center = img(x, y);
                    neighbor(8) = img(x-blockSize, y-blockSize) >= center;
                    neighbor(7) = img(x-blockSize, y  ) >= center;
                    neighbor(6) = img(x-blockSize, y+blockSize) >= center;
                    neighbor(5) = img(x  , y+blockSize) >= center;
                    neighbor(4) = img(x+blockSize, y+blockSize) >= center;
                    neighbor(3) = img(x+blockSize, y  ) >= center;
                    neighbor(2) = img(x+blockSize, y-blockSize) >= center;
                    neighbor(1) = img(x  , y-blockSize) >= center;
                    patt = 0;
                    for k = 1:8
                        patt = patt + neighbor(k) * bitshift(1, k-1);
                    end
                    feature(x, y) = patt;
                end
            end
            
            faceImg = imresize(feature, reSize);
            faceImg = Preprocess(faceImg);
            
            faceImg = faceImg - meanFace;
            feature =  eigenFaces * faceImg;
        end
        
    end
end