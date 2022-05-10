function img = Preprocess(unprocessImg)
    if size(unprocessImg,3) == 3
        greyImg = rgb2gray(unprocessImg);
    else
        greyImg = unprocessImg;
    end
    nolGrey = double(greyImg(:))/255';
    img = (nolGrey - mean(nolGrey)) / std(nolGrey);
end