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
cancer_data = pd.read_csv("breast-cancer-wisconsin.data", sep=",")
cancer_data = cancer_data[["clump_thickness", "uniformity_si", "uniformity_sh", "marginal_adhesion", "epithelial_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "diagnosis"]]
goal_prediction = "diagnosis"

# configuring what we want our data and the prediction goal as numbers to be
X = np.array(cancer_data.drop(goal_prediction, 1))
Y = np.array(cancer_data[goal_prediction])

# splicing up which data is the training and what the final test will be on
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# using kernels to convert values into multiple dimensions and make line more accurate
svm_clf = svm.SVC(kernel="linear", C=1, probability=True)
# slapping the line in
svm_clf.fit(x_train, y_train)

# all of our machine's predictions in an array
y_prediction = svm_clf.predict(x_test)

# accuracy of our hyperplane splitting
accuracy = metrics.accuracy_score(y_test, y_prediction)
print("Accuracy: ", "%.01f" % (accuracy * 100), "%", sep="")

# in our data set, malignant is a value of 2 and benign is a value of 4
# we are replacing them to make it more readible
classes = ["_", "_", "malignant", "_", "benign"]

# run through each part of the array y_predict to visualize the predictions
for x in range(len(y_prediction)):
    print("Predicted: ", classes[y_prediction[x]], " | Data: ", x_test[x], " | Actual: ", classes[y_test[x]], sep="")
