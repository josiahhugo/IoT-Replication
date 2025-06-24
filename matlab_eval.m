% malware_cnn_train.m

% Load exported features and labels
data = load('malware_dataset.mat');
X = data.X;          % [samples x features]
Y = categorical(data.Y);  % 'benign' or 'malware'

% Split data (80% train, 20% test)
cv = cvpartition(Y, 'HoldOut', 0.2);
XTrain = X(training(cv), :);
YTrain = Y(training(cv));
XTest = X(test(cv), :);
YTest = Y(test(cv));

% Define CNN layers
layers = [
    featureInputLayer(size(XTrain, 2), 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.5, 'Name', 'drop1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2, 'Name', 'fc3')  % 2 classes
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict and evaluate
YPred = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Confusion matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

% Optional: Precision, Recall, F1 (custom logic if needed)
