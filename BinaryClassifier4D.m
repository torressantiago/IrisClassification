%% Binary classifier 4D
load fisheriris

% Database creation
Y = ones(150,1);
X = meas;
Y(1:50,:) = 1; % setosa
Y(51:100,:) = 2; % versicolor
Y(101:150,:) = 3; % virginica

%% SVM: fitcsvm
% Classification between versicolor and virginica.
% _Step 1: Remove virginica from database_
inds = ~strcmp(species,'setosa');
Xv1 = meas(inds,:);
Yv1 = Y(inds,:);

% _Step 2: Partition resulting database for cross-validation purposes_
Partition = cvpartition(Yv1,'Holdout',30/100);
TestP = Partition.test;
% Train set
Xv1Train = Xv1(~TestP,:);
Yv1Train = Yv1(~TestP,:);
% Test set
Xv1Test = Xv1(TestP,:);
Yv1Test = Yv1(TestP,:);

% _Step 3: Implement classifier using fitcsvm_
SVMModel = fitcsvm(Xv1Train,Yv1Train);
SupportVect = SVMModel.SupportVectors;

% _Step 4: Obtain performance of classifier_
label = predict(SVMModel,Xv1Test);
% Confusion matrix generation
[C, ~] = confusionmat(Yv1Test,label);
% Cm = confusionchart(Yv1Test,label);

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