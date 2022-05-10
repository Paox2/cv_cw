
classdef classifiarSiameseNetwork
    methods(Static)
        
        %% create network
        
        function net = createNetwork()
            
            layers = [
                imageInputLayer([100 100 3],'Name','input', 'Normalization', 'None');
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
                fullyConnectedLayer(1000,'Name','fc6')];
            lgraph = layerGraph(layers);
            net = dlnetwork(lgraph);
            
%            net = alexnet;
%             net = vgg16;
%            lgraph = layerGraph(net.Layers);
%             lgraph = removeLayers(lgraph, "fc6"); 
%             lgraph = removeLayers(lgraph, "relu6");
%             lgraph = removeLayers(lgraph, "drop6");
%             lgraph = removeLayers(lgraph, "fc7");
%             lgraph = removeLayers(lgraph, "relu7");
%             lgraph = removeLayers(lgraph, "drop7");
%             lgraph = removeLayers(lgraph, "fc8");
%             lgraph = removeLayers(lgraph, "prob");
%            lgraph = removeLayers(lgraph, "output");
            
%            net = dlnetwork(lgraph);
            fprintf("Finish create network\n");
        end
        
        
        %% train
        
        function [fcParams, net] = trainNetwork(net, xTrain, yTrain)
            fcWeights = dlarray(0.01*randn(1,4096));
            fcBias = dlarray(0.01*randn(1,1));

            fcParams = struct(...
                "FcWeights",fcWeights,...
                "FcBias",fcBias);
            
            numIterations = 5000;
            miniBatchSize = 128;
            stopTag = 0;
            
            learningRate = 1e-4;
            gradDecay = 0.9;
            gradDecaySq = 0.99;
            executionEnvironment = "false";
            
            trailingAvgSubnet = [];
            trailingAvgSqSubnet = [];
            trailingAvgParams = [];
            trailingAvgSqParams = [];
            
            figure
            C = colororder;
            lineLossTrain = animatedline('color', C(2,:));
            set(gca, 'YScale', 'log')
            xlabel("Iteration")
            xlim([0 Inf])
            ylabel("Loss")
            grid on
            
            start = tic;
            unImproveNum = 0;
            preAcc = 0;
            
            for iteration = 1:numIterations
                fprintf("\n>>>>>>>>>> Iteration: %d\n", iteration);
                [X1,X2,pairLabels] = classifiarSiameseNetwork.getSiameseBatch(xTrain,yTrain,miniBatchSize);

                % "SSCB" (spatial, spatial, channel, batch) for image data
                X1 = dlarray(X1,"SSCB");
                X2 = dlarray(X2,"SSCB");

                if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                    X1 = gpuArray(X1);
                    X2 = gpuArray(X2);
                end

                [loss,gradientsSubnet] = dlfeval(@classifiarSiameseNetwork.modelLoss2,net,X1,X2,pairLabels);

                [net,trailingAvgSubnet,trailingAvgSqSubnet] = adamupdate(net,gradientsSubnet, ...
                    trailingAvgSubnet,trailingAvgSqSubnet,iteration,learningRate,gradDecay,gradDecaySq);

                
                % plot
                D = duration(0,0,toc(start),"Format", "hh:mm:ss");
                lossValue = double(loss);
                lossValue
                addpoints(lineLossTrain,iteration,extractdata(lossValue));

                title("Elapsed: " + string(D))
                drawnow
                
                if stopTag == 1
                    break
                end
            end
        end
        
        
        %% loss calculation
        
        function [curAcc, loss,gradientsSubnet,gradientsParams] = modelLoss(net,fcParams,X1,X2,pairLabels)
            Y = classifiarSiameseNetwork.forwardSiamese(net,fcParams,X1,X2);

            
            tempY = gather(extractdata(Y));
            tempY = round(tempY);
            curAcc = sum(tempY == pairLabels) / size(pairLabels,2);
            loss = classifiarSiameseNetwork.binarycrossentropy(Y,pairLabels);
            [gradientsSubnet,gradientsParams] = dlgradient(loss,net.Learnables,fcParams);   
        end
        
        function [loss,gradients] = modelLoss2(net,X1,X2,pairLabels)
            F1 = forward(net,X1);
            % Pass second set of image pairs forward through the network
            F2 = forward(net,X2);

            % Calculate contrastive loss
            margin = 0.3;
            loss = classifiarSiameseNetwork.contrastiveLoss(F1,F2,pairLabels,margin);

            % Calculate gradients of the loss with respect to the network learnable
            % parameters
            gradients = dlgradient(loss,net.Learnables);
        end
        
        
%% loss

        function loss = contrastiveLoss(F1,F2,pairLabel,margin)

            % Define small value to prevent taking square root of 0
            delta = 1e-6;
            distances = sqrt(sum((F1 - F2).^2,1) + delta);

            % label(i) = 1 if features1(:,i) and features2(:,i) are features
            % for similar images, and 0 otherwise
            lossSimilar = pairLabel.*(distances.^2);
            lossDissimilar = (1 - pairLabel).*(max(margin - distances, 0).^2);

            loss = 0.5*sum(lossSimilar + lossDissimilar,"all");

        end
        
        
        function loss = binarycrossentropy(Y,pairLabels)
            % Get precision of prediction to prevent errors due to floating point
            % precision.
            precision = underlyingType(Y);

            % Y < eps -> eps.
            % 1 - eps < Y < 1 -> 1 - eps
            Y(Y < eps(precision)) = eps(precision);
            Y(Y > 1 - eps(precision)) = 1 - eps(precision);

            % Calculate binary cross-entropy loss for each pair
            loss = -pairLabels.*log(Y) - (1 - pairLabels).*log(1 - Y);

            % Sum over all pairs in minibatch and normalize.
            loss = sum(loss)/numel(pairLabels);
        end

        
        %% batch
        
        function [X1,X2,pairLabels] = getSiameseBatch(X, Y, miniBatchSize)
            pairLabels = zeros(1,miniBatchSize);
            imgSize = [size(X,1) size(X,2)];
            X1 = zeros([imgSize 3 miniBatchSize],"single");
            X2 = zeros([imgSize 3 miniBatchSize],"single");

            for i = 1:miniBatchSize
                choice = rand(1);

                if choice < 0.5
                    [pairIdx1,pairIdx2,pairLabels(i)] = classifiarSiameseNetwork.getSimilarPair(Y);
                else
                    [pairIdx1,pairIdx2,pairLabels(i)] = classifiarSiameseNetwork.getDissimilarPair(Y);
                end

                X1(:,:,:,i) = X(:,:,:,pairIdx1);
                X2(:,:,:,i) = X(:,:,:,pairIdx2);
            end
        end
        
        
        function [pairIdx1,pairIdx2,pairLabel] = getSimilarPair(classLabel)
            classes = unique(classLabel);
            classChoice = randi(numel(classes));
            idxs = find(classLabel==classes(classChoice));

            % Randomly choose two different images from the chosen class.
            pairIdxChoice = randperm(numel(idxs),2);
            pairIdx1 = idxs(pairIdxChoice(1));
            pairIdx2 = idxs(pairIdxChoice(2));
            pairLabel = 1;
        end
        
        
        function  [pairIdx1,pairIdx2,label] = getDissimilarPair(classLabel)
            classes = unique(classLabel);
            classesChoice = randperm(numel(classes),2);
            idxs1 = find(classLabel==classes(classesChoice(1)));
            idxs2 = find(classLabel==classes(classesChoice(2)));

            % Randomly choose one image from each class.
            pairIdx1Choice = randi(numel(idxs1));
            pairIdx2Choice = randi(numel(idxs2));
            pairIdx1 = idxs1(pairIdx1Choice);
            pairIdx2 = idxs2(pairIdx2Choice);
            label = 0;
        end
        
        
        %% forward predict
        
        function Y = forwardSiamese(net,fcParams,X1,X2)
            Y1 = predict(net,X1);
            Y1 = sigmoid(Y1);

            Y2 = predict(net,X2);
            Y2 = sigmoid(Y2);

            Y = abs(Y1 - Y2);

            % Pass result through a fullyconnect operation.
            Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

            % Convert to probability between 0 and 1.
            Y = sigmoid(Y);
        end       
        
        function Y = predictSiamese(net,fcParams,X1,X2)
            Y1 = predict(net,X1);
            Y1 = sigmoid(Y1);

            Y2 = predict(net,X2);
            Y2 = sigmoid(Y2);
            Y = abs(Y1 - Y2);

            % Pass result through a fullyconnect operation.
            Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

            % Convert to probability between 0 and 1.
            Y = sigmoid(dis);
        end

        
    end
end