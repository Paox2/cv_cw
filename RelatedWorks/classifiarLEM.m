%% bad performance, abandoned
classdef classifiarLEM
    methods(Static)
        
        
        function [owt,w,b] = LEMtrain(featureImgSet, L, m)
            featureImgSet = featureImgSet';
            [n,d] = size(featureImgSet);
            w = rand(d, L) * 2 - 1;
            b = rand(1, L);
            ind = ones(n, 1);
            b1 = b(ind, :);
            
            % get H
            tempH = featureImgSet * w + b1;
            H = classifiarLEM.g(tempH);
            
            tempT = zeros(n,m);
            for i = 1 : n
                tempT(i,i) = 1; %%%%%%%% now: label i is object 'i'
            end
            T = tempT * 2 - 1;
            
            owt = pinv(H) * T;
        end
        
        
        
        function resultID = LEMtest(labels, faceImgTest, owt, w, b)
            faceImgTest = faceImgTest';
            [n,~] = size(faceImgTest);
            ind = ones(n,1);
            b1 = b(ind,:);
            tempH = faceImgTest * w + b1;
            H = classifiarLEM.g(tempH);
            
            similar = H * owt;
            [~,resultID] = max(similar(n,:));
            resultID = labels(resultID,:);
            
        end
        
        
        function H = g(tempH)
            H = 1 ./ (1 + exp(-1 * tempH));
        end
    end
end