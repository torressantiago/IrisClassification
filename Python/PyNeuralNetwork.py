from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import numpy as np
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# This code will implement a Neural Network on the famous Iris dataset
# Load dataset
DataSet = pandas.read_csv('irisData.csv', header = None)

# Separate into features and labels
X = DataSet[[0,1,2,3]]; X = X.to_numpy()
Y = DataSet[[5]]; Y = np.int32(Y.to_numpy()); 

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(Y)

# View histograms of each characteristic
fig, axs = plt.subplots(2, 2)

axs[0, 0].hist(X[0:150,0])
axs[0, 0].set_title('Histogram of charasteristic 1')
axs[0, 0].grid(True)

axs[0, 1].hist(X[0:150,1])
axs[0, 1].set_title('Histogram of charasteristic 2')
axs[0, 1].grid(True)

axs[1, 0].hist(X[0:150,2])
axs[1, 0].set_title('Histogram of charasteristic 3')
axs[1, 0].grid(True)

axs[1, 1].hist(X[0:150,3])
axs[1, 1].set_title('Histogram of charasteristic 4')
axs[1, 1].grid(True)
fig.show()
# Remove characteristic 4 since it's all over the place
X = X[0:150,0:3]

# Plot feature map
fig = plt.figure()
xyz = fig.add_subplot(111, projection='3d')
xyz.scatter(X[0:50,0],X[0:50,1],X[0:50,2])
xyz.scatter(X[50:100,0],X[50:100,1],X[50:100,2])
xyz.scatter(X[100:150,0],X[100:150,1],X[100:150,2])
xyz.view_init(elev=10., azim=100)

xyz.set_xlabel('Characteristic 1')
xyz.set_ylabel('Characteristic 2')
xyz.set_zlabel('Characteristic 3')

xyz.legend(['Setosa','Virginica','Versicolor'])

fig.show()

# Separate data into a training set and a test set
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size=0.3)

# Build a multiclass classifier using a multilayer perceptron neural network
clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(20, 20), random_state=1, max_iter = 100000000)

# Fit model with training data
Classifier = clf.fit(XTrain, YTrain)

# Test classifier
PredictedLabels = clf.predict(XTest)

# Observe performance of classifier
ConfusionM = confusion_matrix(YTest.argmax(axis=1), PredictedLabels.argmax(axis=1))
Performance = classification_report(YTest,PredictedLabels)

print(ConfusionM)
print(Performance)
