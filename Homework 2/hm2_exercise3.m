%% Homework 2: Classification challenge
% Albin Ekström
% Date 2 okt 2022

clc
clear variables

% Load MNIST training data
[xTrain, tTrain, xVal, tVal, xTest, tTest] = LoadMNIST(3);

%% NETWORK SETUP
% Structure taken from:
% "Create Simple Deep Learning Network for Classification"
clc

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'Momentum',0.9,...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'MiniBatchSize',64, ...
    'ValidationData',{xVal tVal}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',5,...
    'OutputNetwork','best-validation-loss',...
    'Verbose',false, ...
    'Plots','training-progress');

%% TRAINING NETWORK
net = trainNetwork(xTrain,tTrain,layers,options);

%% TESTING TRAINING DATA
tPred = classify(net,xTest);
accuracy = sum(tPred == tTest)/numel(tTest)

%% TEST FINAL DATA
% Load mnist data
finalTest = loadmnist2(); 
tPred_final = classify(net,finalTest);

%% EXPORT CSV
csvwrite('CSV/classifications.csv',str2num(char(tPred_final)))

%% PLOT
n = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    colormap(gray(256))
    image(finalTest(:,:,:,n(i)))
    set(gca,'XTick',[], 'YTick', [])
    title("Predicted nbr: " + str2num(char(tPred_final(n(i)))))
end
