classdef classifiarBP
    methods(Static)
        
        function net = BPtrain(featureImgSet, BPparameter, trainLabel)
            [d, n] = size(featureImgSet);
            featureImgSet = featureImgSet';
            P = mapminmax(featureImgSet);
            
            classLabel = unique(trainLabel);
            nClass = length(classLabel);
            trueValue = zeros(n,nClass);
            for i = 1 : n
                trueValue(i,trainLabel(i)) = 1.0;
            end
            
            % distort the order
            trainSet(:,1:d) = P;
            trainSet(:,(d+1):(d+nClass)) = trueValue;
            permTrainSet = trainSet(randperm(n),:);
            P = permTrainSet(:,1:d)';
            trueValue = permTrainSet(:,(d+1):(d+nClass))';
            
            net = newcf(minmax(P), trueValue, [BPparameter.hideN,nClass],...
                {BPparameter.fIn, BPparameter.fOut}, BPparameter.fTrain);
            net.trainParam.epochs = BPparameter.epochs;
            net.trainParam.goal = BPparameter.goal;
            net.trainParam.lr = BPparameter.lr;
            net.divideFcn = BPparameter.divideFcn;
            net = train(net, P, trueValue);
            
        end
        
        
        
        function resultID = BPtest(labels, faceImgTest, net)
            faceImgTest = faceImgTest';
            P = mapminmax(faceImgTest);
            
            Y = sim(net, P');
            [~, indice] = sort(Y);
            
            index = indice(size(labels,1),1);
            resultID = labels(index,:);
            
        end
        
    end
end