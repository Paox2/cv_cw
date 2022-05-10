%% CNN
classdef classifiarCNN
    methods(Static)
        
        function lgraph = createNet(n)
            layers = [
                imageInputLayer([100 100 3],'Name','input');
                convolution2dLayer(5,32,'Stride',1,'Name','conv1');
                reluLayer('Name','relu1');
                layerNormalizationLayer('Name','norm1');
                maxPooling2dLayer(2,'Stride',2,'Name','pool1');
                convolution2dLayer(5, 64, 'Stride', 1,'Name','conv2');
                reluLayer('Name','relu2'); 
                layerNormalizationLayer('Name','norm2');
                maxPooling2dLayer(2, 'Stride', 2,'Name','pool2');
                convolution2dLayer(3,128,'Stride',1,'Name','conv3');
                reluLayer('Name','relu3');
                layerNormalizationLayer('Name','norm3');
                maxPooling2dLayer(2,'Stride', 2,'Name','pool3');
                convolution2dLayer(3, 256, 'Stride', 1,'Name','conv4');
                reluLayer('Name','relu4');
                maxPooling2dLayer(2, 'Stride', 2,'Name','pool4');
                fullyConnectedLayer(2048,'Name','fc15');
                dropoutLayer('Name','drop5');
                fullyConnectedLayer(n,'Name','fc6');
                softmaxLayer('Name','sm6')
                classificationLayer('Name','output')];
            lgraph = layerGraph(layers);
            
%             newConnectedLayer = fullyConnectedLayer(n,'Name','new_fc',...
%                 'WeightLearnRateFactor',n,'BiasLearnRateFactor',n);
%             newClassLayer = classificationLayer('Name','new_classoutput');
%             
%             % alexnet
%             net = alexnet;
%             lgraph = layerGraph(net.Layers);
%             lgraph = replaceLayer(lgraph,'fc8',newConnectedLayer);
%             lgraph = replaceLayer(lgraph,'output',newClassLayer);


%             % vgg16
%             net = vgg16;
%             lgraph = layerGraph(net.Layers);
%             lgraph = replaceLayer(lgraph, 'fc8', newConnectedLayer);
%             lgraph = replaceLayer(lgraph, 'output', newClassLayer);
            
        end
        
        
        function net = trainCNN(xTrain, yTrain, lgraph, opts)
            [net,~] = trainNetwork(xTrain, yTrain, lgraph, opts);
        end
        
        
        function resultID = testCNN(testImg, net)
            resultID = classify(net,testImg);
        end
        
    end
end