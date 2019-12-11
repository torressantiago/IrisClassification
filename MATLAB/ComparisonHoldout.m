%% Comparison in between cases using holdout separation mehtod
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

% _Step 2: Partition resulting database for SVM classification_
Partition = cvpartition(Yv1,'Holdout',30/100);
TestP = Partition.test;
% Train set
Xv1Train = Xv1(~TestP,:);
Yv1Train = Yv1(~TestP,:);
% Test set
Xv1Test = Xv1(TestP,:);
Yv1Test = Yv1(TestP,:);

% _Step 3: Partition resulting database for ANN classification_
% New Y definition
YNN = ones(150,3);

YNN(1:100,1) = zeros(100,1);
YNN(1:50,2) = zeros(50,1);
YNN(101:150,2) = zeros(50,1);
YNN(51:150,3) = zeros(100,1);

% Train set
Yv2Train = YNN(~TestP,:);
% Test set
Yv2Test = YNN(TestP,:);
Yv2Testa = Yv1(TestP,:);
%% NN: feedforwardnet
% _Step 1: Implement classifier using feedforwardnet_
net = feedforwardnet([8 8]);

net.trainParam.epochs = 1000;

[net, tr] = train(net,Xv1Train',Yv2Train');
view(net)

% _Step 2: Obtain classifier performance_
label = net(Xv1Test');
label = label';
labela = label;
labelb = ones(45,1);

for i = 1:45
    for j = 1:3
        if labela(i,j) < 0.5
            labela(i,j) = 0;
        else
            labela(i,j) = 1;
        end
    end
end

LUT = [0,0,0 ; 0,0,1 ; 0,1,0 ; 0,1,1 ; 1,0,0 ; 1,0,1 ; 1,1,0 ; 1,1,1];

for i = 1:45
    if isequal(labela(i,:), LUT(1,:)) || isequal(labela(i,:), LUT(2,:))
        labelb(i) = 1;
    elseif isequal(labela(i,:), LUT(3,:)) || isequal(labela(i,:), LUT(4,:))
        labelb(i) = 2;
    elseif isequal(labela(i,:), LUT(5,:)) || isequal(labela(i,:), LUT(6,:))
        labelb(i) = 3;
    else 
        labelb(i) = 4;
    end
end

% Confusion matrix generation
[CNN, ~] = confusionmat(Yv2Testa,labelb);
figure
title('Confusion chart for ANN')
cm = confusionchart(Yv2Testa,labelb);
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