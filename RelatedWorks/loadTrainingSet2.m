function [imgSet, personID]=loadTrainingSet2(imgPath, useAll)
% imgSet stores the training images
% personID is the corresponding ID for each image. 

folderNames=dir(imgPath);
imgSet=[]; % all images are 3 channels with size of 600x600
personID=[]; % the folder names are the labels
k=1;
for i=5:length(folderNames)
    imgName=dir([imgPath, folderNames(i,:).name,'\*.jpg']);
    if length(imgName) > 1 || useAll == 1
        for j = 1 : length(imgName)
            imgSet(:,:,:,k)= imread([imgPath, folderNames(i,:).name, '\', imgName(j).name]);
            personID=[personID, i-4];  %the folder names are the persons IDs. 
            k=k+1;
        end
    end
end
imgSet=uint8(imgSet(:,:,:,1:k-1));   % Note that it is in unsigned integer 8 format with intensity range of 0 to 255.
%figure,imshow(trainImgSet(:,:,:,1)) % check the first image. 