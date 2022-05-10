function faceImages = faceDetection(oriImageSet, outputSize)
    faceDetector = vision.CascadeObjectDetector();
    nFalse = 0;
    oh = size(oriImageSet,1);
    ow = size(oriImageSet,2);
    lFaces = zeros(size(oriImageSet,4),4);
    for i = 1:size(oriImageSet,4)   
        lFace = faceDetector(oriImageSet(:,:,:,i));
        if isempty(lFace)
            lFace = [1 1 ow oh];
            nFalse = nFalse + 1;
        else
            detectionSize = lFace(:,4) .* lFace(:,3);
            facePos = detectionSize == max(detectionSize);
            lFace = lFace(facePos,:);
        end
        lFaces(i,:) = lFace(1,:);
    end
    
    % average number remove uncognition face image
    aveH = round(mean(lFaces(lFaces(:,4)~=oh, 4)));
    aveW = round(mean(lFaces(lFaces(:,3)~=ow, 3)));
    
    if isempty(outputSize)
        faceImages = zeros(aveH, aveW, 3, size(oriImageSet,4), 'uint8');
    else
        faceImages = zeros(outputSize(1), outputSize(2), 3, size(oriImageSet,4), 'uint8');
    end
    
    for i = 1:size(oriImageSet,4)
        faceImage = imcrop(oriImageSet(:,:,:,i), lFaces(i,:));
        if isempty(outputSize)
            faceImages(:,:,:,i) = imresize(faceImage, [aveH aveW]);
        else
            faceImages(:,:,:,i) = imresize(faceImage, outputSize);
        end
    end
end