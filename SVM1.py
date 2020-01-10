import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#video time stamp 01:31
#SVM stands for support veector machine

cancer = datasets.load_breast_cancer()
print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target
#the number in the back is how much % data you want to test with
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train,y_train)


#c= 0 : hard margin , c=1 : soft margin , c=2 double as soft ( just more datapoints )
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)