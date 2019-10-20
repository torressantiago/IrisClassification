%% Neural Network implementation
load fisheriris

% Database creation
Y = ones(150,1);
X = meas;
Y(1:50,:) = 1; % setosa
Y(51:100,:) = 2; % versicolor
Y(101:150,:) = 3; % virginica

%% NN: feedforwardnet
% _Step 1: Create database for problem_
Xv1 = meas;
Yv1 = Y;

% _Step 2: Partition resulting database for cross-validation purposes_
Partition = cvpartition(Yv1,'Holdout',70/100);
TestP = Partition.test;
% Train set
Xv1Train = Xv1(~TestP,:);
Yv1Train = Yv1(~TestP,:);
% Test set
Xv1Test = Xv1(TestP,:);
Yv1Test = Yv1(TestP,:);

% _Step 3: Implement classifier using feedforwardnet_
net = feedforwardnet([ 8 8 ]);

net.trainParam.epochs = 1000;

[net, tr] = train(net,Xv1Train',Yv1Train');
view(net)

% _Step 4: Obtain performance of classifier_
label = net(Xv1Test');
label = label';
labela(label < 1.5 & label >= 0.5) = 1;
labela(label < 2.5 & label >= 1.5) = 2;
labela(label < 3.5 & label >= 2.5) = 3;
% Confusion matrix generation
[C, ~] = confusionmat(Yv1Test,labela);