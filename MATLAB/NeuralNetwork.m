%% Neural Network implementation
load fisheriris

% Database creation
Y = ones(150,3);
X = meas;
% Y(1:50,:) = [0,0,1]; % setosa
% Y(51:100,:) = [0,1,0]; % versicolor
% Y(101:150,:) = [1,0,0]; % virginica

Y(1:100,1) = zeros(100,1);
Y(1:50,2) = zeros(50,1);
Y(101:150,2) = zeros(50,1);
Y(51:150,3) = zeros(100,1);

Yh(1:50,:) = 1; % setosa
Yh(51:100,:) = 2; % versicolor
Yh(101:150,:) = 3; % virginica

%% NN: feedforwardnet
% _Step 1: Create database for problem_
Xv1 = meas;
Yv1 = Y;

% _Step 2: Partition resulting database for cross-validation purposes_
Partition = cvpartition(Yh,'Holdout',30/100);
TestP = Partition.test;
% Train set
Xv1Train = Xv1(~TestP,:);
Yv1Train = Yv1(~TestP,:);
% Test set
Xv1Test = Xv1(TestP,:);
Yv1Test = Yv1(TestP,:);
Yv1Testa = Yh(TestP,:);

% _Step 3: Implement classifier using feedforwardnet_
net = feedforwardnet([8 8]);

net.trainParam.epochs = 1000;

[net, tr] = train(net,Xv1Train',Yv1Train');
view(net)

% _Step 4: Obtain performance of classifier_
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
[C, ~] = confusionmat(Yv1Testa,labelb);