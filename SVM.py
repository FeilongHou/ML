# SVM support vector machine
# 2 closest point to hyperplane from each class 
# have max distance
# doing a kernel will give high dimension data

#between support vectors, we can have a thin margin that allow outlier to exist
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)

X = cancer.data
Y = cancer.target   # maglignant or benign

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, train_size = 0.2)

classes = ["malignant", "benign"]

# classfier: support vector classification
# try to see without using kernel
clf = svm.SVC(kernel="linear", C = 2) # C is the soft margin =0 meanning no margin smaller C bigger margin
#clf = svm.SVC(kernel="poly", degree = 2)
clf.fit(x_train, y_train)

# predict data before score them
y_pred = clf.predict(x_test)
# use metrix to find score
acc = metrics.accuracy_score(y_test, y_pred)
prec = metrics.precision_score(y_test,y_pred)
print("Precision: " + str(prec))
print(acc)
 

#compare to K nearest neighbor
clf2 = KNeighborsClassifier(n_neighbors=13)
clf2.fit(x_train, y_train)
acc = clf2.score(x_test,y_test)
print(acc)

y_pred = clf2.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)