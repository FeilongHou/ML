# panda read first line of fine as attibute
# this is K nearest neightbor
# K is hyper parameter, the amount of neightbors
# K = 3, find the nearest 3 points to the prediction point then vote
# K needs to be an odd number
# Dont pick too large of K
# Computational heavy
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

# This will change label from str to int
sort = preprocessing.LabelEncoder()
# Get buying column and turn into a list then transfer to appropiate integer value
buying = sort.fit_transform(list(data["buying"]))
maint = sort.fit_transform(list(data["maint"]))
door = sort.fit_transform(list(data["door"]))
persons = sort.fit_transform(list(data["persons"]))
lug_boot = sort.fit_transform(list(data["lug_boot"]))
safety = sort.fit_transform(list(data["safety"]))
cla = sort.fit_transform(list(data["class"]))

preditc = "class"

# combine all the numerical attibute into one list
# zip() convert numpy array into a list
X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cla)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors= 11)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

prediction = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]   # correspounding to numerical valuesi in data(0-3)

for i in range(len(x_test)):
    print("Pridiced:", names[prediction[i]], " Data: ", x_test[i], " Actual:", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 11, True)  # it needs to be a 2D array returns 2 arrays one is distance, the second is index
    print("N:", n)                               # returns a distance array and index of neightbot
