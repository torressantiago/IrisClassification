%% Comparison in between cases using cross validation separation mehtod
load fisheriris

% Database creation
Y = ones(150,1);
X = meas;
Y(1:50,:) = 1; % setosa
Y(51:100,:) = 2; % versicolor
Y(101:150,:) = 3; % virginica

%% NN: feedforwardnet
% Classification between versicolor and virginica.
% _Step 1: Create database for problem
Xv1 = meas;
Yv1 = Y;

% _Step 2: Partition resulting database for cross-validation purposes_
Partition = cvpartition(Yv1,'Holdout',30/100);
TestP = Partition.test;
% Train set
Xv1Train = Xv1(~TestP,:);
Yv1Train = Yv1(~TestP,:);
% Test set
Xv1Test = Xv1(TestP,:);
Yv1Test = Yv1(TestP,:);

% K folds
KPartition = cvpartition(Yv1Train,'kFold',10);
for i = 1:KPartition.NumTestSets
    KTrain(:,i) = KPartition.training(i);
    KTest(:,i) = KPartition.test(i);
end

% _Step 3: Implement classifier using feedforwarnet_
net = feedforwardnet([3 3]);
net.trainParam.epochs = 1000;
% New train set
for i = 1:KPartition.NumTestSets
    XTrain = Xv1Train(~KTest(:,i),:);
    YTrain = Yv1Train(~KTest(:,i),:);
    if i == 1
        [net1, ~] = train(net,XTrain',YTrain');
    elseif i == 2
        [net2, ~] = train(net,XTrain',YTrain');
    elseif i == 3
        [net3, ~] = train(net,XTrain',YTrain');
    elseif i == 4
        [net4, ~] = train(net,XTrain',YTrain');
    elseif i == 5
        [net5, ~] = train(net,XTrain',YTrain');
    elseif i == 6
        [net6, ~] = train(net,XTrain',YTrain');
    elseif i == 7
        [net7, ~] = train(net,XTrain',YTrain');
    elseif i == 8
        [net8, ~] = train(net,XTrain',YTrain');
    elseif i == 9
        [net9, ~] = train(net,XTrain',YTrain');
    elseif i == 10
        [net10, ~] = train(net,XTrain',YTrain');
    end
end

% _Step 4: Obtain performance of classifier_
for i = 1:KPartition.NumTestSets
    if i == 1
        temp1 = net1(Xv1Test');
        label1 = temp1';
    elseif i == 2
        temp2 = net1(Xv1Test');
        label2 = temp2';
    elseif i == 3
        temp3 = net1(Xv1Test');
        label3 = temp3';
    elseif i == 4
        temp4 = net1(Xv1Test');
        label4 = temp4';
    elseif i == 5
        temp5 = net1(Xv1Test');
        label5 = temp5';
    elseif i == 6
        temp6 = net1(Xv1Test');
        label6 = temp6';
    elseif i == 7
        temp7 = net1(Xv1Test');
        label7 = temp7';
    elseif i == 8
        temp8 = net1(Xv1Test');
        label8 = temp8';
    elseif i == 9
        temp9 = net1(Xv1Test');
        label9 = temp9';
    elseif i == 10
        temp10 = net1(Xv1Test');
        label10 = temp10';
    end
end

% Confusion matrix generation
[C, ~] = confusionmat(Yv1Test,label);
Cm = confusionchart(Yv1Test,label);

% These values hold for a 2x2 matrix confusion. In order to observe the
% performance of a certain characteristic given the characteristic itself
% or another one, the calculations must be adjusted.
TP = C(1,1); FP = C(2,1); FN = C(1,2); TN = C(2,2);
All = TP + TN; P = TP + FN; N = FP + TN; Pp = TP + FP; Np = FN + TN;
Accuracy = (TP+TN)/All;
ErrorRate = (FP+FN)/All;
Sensitivity = TP/P;
Specificity = TN/N;
Precision = TP/(TP+FP);
Recall = TP/(TP+FP);
FScore = (2*Precision*Recall)/(Precision+Recall);

Mperformance = table(Accuracy, ErrorRate, Sensitivity, Specificity, Precision,...
    Recall, FScore);