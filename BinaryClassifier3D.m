%% Binary classifier 3D
load fisheriris

% Database creation
Y = ones(150,1);
X = meas;
Y(1:50,:) = 1; % setosa
Y(51:100,:) = 2; % versicolor
Y(101:150,:) = 3; % virginica

%% SVM: fitcsvm
% Classification between setosa and versicolor. Will only use
% characteristics 2, 3 and 4
% _Step 1: Remove virginica from database_
inds = ~strcmp(species,'virginica');
Xv1 = meas(inds,2:4);
Yv1 = Y(inds,:);

% _Step 2: Partition resulting database for cross-validation purposes_
Partition = cvpartition(Yv1,'Holdout',25/100);
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
scatter3(Xv1Train(Yv1Train==1,1),Xv1Train(Yv1Train==1,2),Xv1Train(Yv1Train==1,3))
hold on
scatter3(Xv1Train(Yv1Train==2,1),Xv1Train(Yv1Train==2,2),Xv1Train(Yv1Train==2,3))
hold on
plot3(SupportVect(:,1),SupportVect(:,2),SupportVect(:,3),'ko','MarkerSize',10);
hold on
%Hyperplane
svm_3d_plot(SVMModel,Xv1Train,Yv1Train);
xlabel('Characteristic 2')
ylabel('Characteristic 3')
zlabel('Characteristic 4')
title('Results from classification')
legend('Setosa','Versicolor','Support vector','Hyperplane')


% _Step 5 : Obtain performance of classifier_
label = predict(SVMModel,Xv1Test);
% Confusion matrix generation
[C, ~] = confusionmat(Yv1Test,label);
% Cm = confusionchart(Yv1Test,label);

% Performance data
% +------------------------------+---------------------------------------+----------------------+-----+
% |                              | C1                                    | C2                   |     |
% | Actual class\Predicted class |                                       |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% |                              | True positives (TP)                   |                      |     |
% | C1                           |                                       | False negatives (FN) | P   |
% +------------------------------+---------------------------------------+----------------------+-----+
% | C2                           | False positives (FP)                  | True negatives (TN)  |     |
% |                              |                                       |                      | N   |
% +------------------------------+---------------------------------------+----------------------+-----+
% |                              | P'                                    | N'                   | All |
% +------------------------------+---------------------------------------+----------------------+-----+
% | Accuracy                     |                                       |                      |     |
% |                              | (TP+TN)/ALL                           |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% | Error rate                   | (FP+FN)/ALL                           |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% | Sensitivity                  |                                       |                      |     |
% |                              | TP/P                                  |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% | Specificity                  | TN/N                                  |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% | Precision                    |                                       |                      |     |
% |                              | TP/(TP+FP)                            |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% | Recall                       | TP/(TP+FP)                            |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+
% | F-score                      | (2*precision*recall)/precision+recall |                      |     |
% +------------------------------+---------------------------------------+----------------------+-----+

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

function [] = svm_3d_plot(mdl,X,group)
     %Gather support vectors from ClassificationSVM struct
     sv =  mdl.SupportVectors;
     %set step size for finer sampling
     d =0.05;
     %generate grid for predictions at finer sample rate
     [x, y, z] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
        min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
     xGrid = [x(:),y(:),z(:)];
     %get scores, f
     [ ~ , f] = predict(mdl,xGrid);
     %reshape to same grid size as the input
     f = reshape(f(:,2), size(x));
     % Assume class labels are 1 and 0 and convert to logical
     t = logical(group);
     %plot decision surface
     [faces,verts,~] = isosurface(x, y, z, f, 0, x);
     patch('Vertices', verts, 'Faces', faces, 'FaceColor','k','edgecolor',... 
     'none', 'FaceAlpha', 0.2);
     grid on
     box on
     view(3)
     hold off
 end