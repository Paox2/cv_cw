classdef featureKPCA
    methods(Static)

        function [kpcaModel, trainFeatureSet] = extractFeatureSet(faceImgs, kpcaModel)
            % faceImgs: featureNumber*imageNumber double matrix
            
            trainSet = zeros(size(faceImgs,1)*size(faceImgs,2), size(faceImgs,4));
            for i=1:size(faceImgs,4)  
                trainSet(:,i) = Preprocess( faceImgs(:,:,:,i));       
            end
            
            
            trainSet = trainSet';
            N = size(trainSet,1);
            
            if strcmp(kpcaModel.kernelType,'gaussian')==1
                DIST = featureKPCA.distanceMatrix(trainSet);
                DIST(DIST==0) = inf;
                DIST = min(DIST);
                kpcaModel.kernelArgs(1) = 5*mean(DIST);
            else
                kpcaModel.kernelArgs = kpcaModel.kernelArgs;
            end
            
            %kernel pca
            k0 = featureKPCA.computeKernel(trainSet,trainSet,kpcaModel.kernelType, kpcaModel.kernelArgs);
            oneN = ones(N,N)/N;
            k = k0 - oneN*k0 - k0*oneN + oneN*k0*oneN;
            
            % eigenvalue analysis
            [V,D] = eig(k);
            eigValue = real(diag(D));
            [~,IX] = sort(eigValue, 'descend');
            eigVector = V(:,IX);
            eigValue = eigValue(IX);
            
            for i = 1 : N
                eigVector(:,i) = eigVector(:,i) / (sqrt(eigValue(i)));
            end
            
            % dimensionality reduction, 90% contribution
            dSumContribution = sum(eigValue);
            dSumExtract = 0;
            newD = 0;
            while (dSumExtract / dSumContribution < 0.98)
                newD = newD + 1;
                dSumExtract = dSumExtract + eigValue(newD);
            end
            
            eigVector = eigVector(:,1:newD);
            trainFeatureSet = k0 * eigVector;
            trainFeatureSet = trainFeatureSet';
            
            kpcaModel.eigVector = eigVector;
            kpcaModel.k = k;
            kpcaModel.oneN = oneN;
            kpcaModel.kernelArgs = kpcaModel.kernelArgs;
            kpcaModel.trainSet = trainSet;
  
        end
    
        
        function feature = extractFeature(kpcaModel, faceImg)
            
            faceImg = faceImg';
            trainSet = kpcaModel.trainSet;
            k0 = featureKPCA.computeKernel(trainSet, faceImg,...
                kpcaModel.kernelType, kpcaModel.kernelArgs);
            oneN = ones(1,size(trainSet,1))/size(trainSet,1);
            k = k0 - oneN * kpcaModel.k - k0 * kpcaModel.oneN ...
                + oneN * kpcaModel.k * kpcaModel.oneN;
            feature = k * kpcaModel.eigVector;
            feature = feature';
        end
        
        
        function kMat = computeKernel(X, Y, kernelType, kernelArgs)
            kMat = [];
            N = size(X,1);
            
            if strcmp(kernelType,'simple')==1
                kMat = Y*X';
            elseif strcmp(kernelType,'poly')==1
                kMat = (Y*X' + kernelArgs(1)).^(kernelArgs(2));
            elseif strcmp(kernelType,'fpp')==1
                kMat = sign(Y*X'+kernelArgs(1)).*((abs(Y*X'+kernelArgs(1))).^(kernelArgs(2)));
            elseif strcmp(kernelType,'tanh')==1
                kMat = tanh(Y*X'+kernelArgs(1));
            elseif strcmp(kernelType,'gaussian')==1
                kMat = featureKPCA.distanceMatrix([X;Y]);
                kMat = kMat(N+1:end,1:N);
                kMat = kMat .^ 2;
                kMat = exp(-kMat ./ (2 * kernelArgs(1) .^ 2));
            end
        end
        
        
        % same as em distance in baselineModel.m, but use another way 
        function D = distanceMatrix(m)
            % m - number * dimension
            N = size(m,1);
            
            mm = sum(m.*m,2);
            mm1 = repmat(mm,1,N);
            mm2 = repmat(mm',N,1);
            
            D = mm1 + mm2 - 2 * (m * m');
            D(D<0) = 0;
        end
            
        
        
    end
end