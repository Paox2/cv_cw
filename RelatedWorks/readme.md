

dataAug/Preprocess:
faceDetection.m
Preprocess // not all method use that


feature extractor:
featureHOG.m    // hog
featurePCA.m    // pca
featuraLDA.m    // pca+lda
featureLBP.m    // pca+lbp
feature2DPCA.m  // 2DPCA, 100*100 image -> 90*100 image 90%
featureKPCA.m   // kernel PCA


classifiar:
baselineModel.m // contains CV, euclidean distance, svm, bayes
classifiarBP.m  // BP network
classifiarLEM.m // LEM
classifiarCNN.m // CNN directly get result


classifiarSiameseNetwork // siamese network, classify or feature extractor


tune:
tuneParamBP.m   // goal, lr, trainscg/traingdx
tuneCNN.m       // hyperparameter
tuneParamSiameseNetwork.m  // tutor provide dataset
tuneParamSiameseNetwork2.m // lfw dataset


use:
useSNN  // try siameseNetwork feature extractor


util:
splitmat.m // sometimes saved model cannot open