# Machine learning using Support Vector Machines using Tensor Libraries
# Splices up classes using a hyperplane in any dimension
# ---Credit---
# Dr. WIlliam H. Wolberg (physician), University of Wisconsin Hospitals, Madison, Wisconsin, USA
# UCI Repository: [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)]

import sklearn
from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd

# loading in our .data file as a csv using pandas
cancerData = pd.read_csv("breast-cancer-wisconsin.data", sep=",")
cancerData = cancerData[["clump_thickness", "uniformity_si", "uniformity_sh", "marginal_adhesion", "epithelial_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "diagnosis"]]
goalPrediction = "diagnosis"

# configuring what we want our data and the prediction goal as numbers to be
X = np.array(cancerData.drop(goalPrediction, 1))
Y = np.array(cancerData[goalPrediction])

# splicing up which data is the training and what the final test will be on
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# using kernels to convert values into multiple dimensions and make line more accurate
svmClf = svm.SVC(kernel="linear", C=1, probability=True)
# slapping the line in
svmClf.fit(x_train, y_train)

# all of our predictions in an array
y_prediction = svmClf.predict(x_test)

# accuracy of our hyperplane splitting
accuracy = metrics.accuracy_score(y_test, y_prediction)
print(accuracy)

# in our data set, malignant is a value of 2 and benign is a value of 4
# we are replacing them to make it more readible
classes = ["_", "_", "malignant", "_", "benign"]

# run through each part of the array y_predict to visualize the predictions
for x in range(len(y_prediction)):
    print("Predicted: ", classes[y_prediction[x]], " | Data: ", x_test[x], " | Actual: ", classes[y_test[x]], sep="")
