# Machine learning using Support Vector Machines
# Splices up classes using a hyperplane in any dimension

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# load a common database to see whether or not a tumor is benign or malignant
cancerData = datasets.load_breast_cancer()

X = cancerData.data
Y = cancerData.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)


# using kernels to convert values into multiple dimensions and make line more accurate
svmClf = svm.SVC(kernel="linear", C=1, probability=True)
# slapping the line in
svmClf.fit(x_train, y_train)

y_prediction = svmClf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)
print(accuracy)

classes = ["malignant", "benign"]

for x in range(len(y_prediction)):
    print("Predicted: ", classes[y_prediction[x]], " | Data: ", x_test[x], " | Actual: ", classes[y_test[x]], sep="")
