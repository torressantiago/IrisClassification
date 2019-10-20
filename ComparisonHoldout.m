%% Comparison in between cases%% 
load fisheriris

% Database creation
Y = ones(150,1);
X = meas;
Y(1:50,:) = 1; % setosa
Y(51:100,:) = 2; % versicolor
Y(101:150,:) = 3; % virginica

% _Step 1: Create database for problem_
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

%% NN: feedforwardnet
% _Step 1: Implement classifier using feedforwardnet_
net = feedforwardnet([ 8 8 ]);

net.trainParam.epochs = 1000;

[net, tr] = train(net,Xv1Train',Yv1Train');
view(net)

% _Step 2: Obtain performance of classifier_
label = net(Xv1Test');
label = label';
labela(label < 1.5 & label >= 0.5) = 1;
labela(label < 2.5 & label >= 1.5) = 2;
labela(label < 3.5 & label >= 2.5) = 3;
% Confusion matrix generation
[CNN, ~] = confusionmat(Yv1Test,labela);
figure
title('Confusion chart for ANN')
cm = confusionchart(Yv1Test,labela);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Confusion chart for ANN';

%% SVM: fitcecoc 1v1
% _Step 1: Implement classifier using fitcecoc_
t = templateSVM('KernelFunction','gaussian');
Model = fitcecoc(Xv1Train,Yv1Train,'Learners',t,'Coding','onevsone'); % One vs. One

% _Step 2: Obtain performance of classifier_
label = predict(Model,Xv1Test);
% Confusion matrix generation
[CSVM1v1, ~] = confusionmat(Yv1Test,label);
figure
title('Confusion chart for SVM with 1v1 gaussian kernel')
cm = confusionchart(Yv1Test,label);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Confusion chart for SVM with 1v1 gaussian kernel';

%% SVM: fitcecoc 1vall
% _Step 1: Implement classifier using fitcecoc_
t = templateSVM('KernelFunction','gaussian');
Model = fitcecoc(Xv1Train,Yv1Train,'Learners',t,'Coding','onevsall'); % One vs. One

% _Step 2: Obtain performance of classifier_
label = predict(Model,Xv1Test);
% Confusion matrix generation
[CSVM1vall, ~] = confusionmat(Yv1Test,label);
figure
title('Confusion chart for SVM with 1vAll gaussian kernel')
cm = confusionchart(Yv1Test,label);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Confusion chart for SVM with 1vAll gaussian kernel';

%% Compute performance for each class
% Sensitivity for setosa
SensitivitySetosa = [CNN(1,1)/(CNN(1,1)+CNN(2,1)+CNN(3,1)),CSVM1v1(1,1)/(CSVM1v1(1,1)+CSVM1v1(2,1)+CSVM1v1(3,1)),CSVM1vall(1,1)/(CSVM1vall(1,1)+CSVM1vall(2,1)+CSVM1vall(3,1))];

% Sensitivity for versicolor
SensitivityVersicolor = [CNN(2,2)/(CNN(1,2)+CNN(2,2)+CNN(3,2)),CSVM1v1(2,2)/(CSVM1v1(1,2)+CSVM1v1(2,2)+CSVM1v1(3,2)),CSVM1vall(2,2)/(CSVM1vall(1,2)+CSVM1vall(2,2)+CSVM1vall(3,2))];

% Sensitivity for virginica
SensitivityVirginica = [CNN(3,3)/(CNN(1,3)+CNN(2,3)+CNN(3,3)),CSVM1v1(3,3)/(CSVM1v1(1,3)+CSVM1v1(2,3)+CSVM1v1(3,3)),CSVM1vall(3,3)/(CSVM1vall(1,3)+CSVM1vall(2,3)+CSVM1vall(3,3))];

% Accuracy
Accuracy = [sum(diag(CNN))/sum(sum(CNN)), sum(diag(CSVM1v1))/sum(sum(CSVM1v1)), sum(diag(CSVM1vall))/sum(sum(CSVM1vall))];