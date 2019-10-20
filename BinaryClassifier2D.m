%% Binary classifier 2D
load fisheriris

% Database creation
Y = ones(150,1);
X = meas;
Y(1:50,:) = 1; % setosa
Y(51:100,:) = 2; % versicolor
Y(101:150,:) = 3; % virginica

%% SVM: fitcsvm
% Classification between setosa and versicolor. Will only use
% characteristics 3 and 4
% _Step 1: Remove virginica from database_
inds = ~strcmp(species,'virginica');
Xv1 = meas(inds,3:4);
Yv1 = species(inds,:);

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

% _Step 4: Observe results_
figure
% Feature space
gscatter(Xv1Train(:,1),Xv1Train(:,2),Yv1Train)
hold on
plot(SupportVect(:,1),SupportVect(:,2),'ko','MarkerSize',10)

%Hyperplane
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(Xv1Train(:,1)):d:max(Xv1Train(:,1)),...
    min(Xv1Train(:,2)):d:max(Xv1Train(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(SVMModel,xGrid);
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
grid on 
legend('Setosa','Versicolor','Support vector','Hyperplane')
hold off
xlabel('Characteristic 3')
ylabel('Characteristic 4')
title('Results from classification')

% _Step 5 : Obtain performance of classifier_
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