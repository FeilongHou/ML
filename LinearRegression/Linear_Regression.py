import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep = ";") # seperator = ;, original data seperated by ;

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))       # return a new data frame what has no G3, this is our trainning data
Y = np.array(data[predict])                 # testing set

x_train, x_test, y_train, y_test =  sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# if we want to train a better model, use the following loop to do so
"""
best = 0
for _ in range(30):
    #we are spliting X&Y into 4 different array
    # x_train is from part of X
    # y_train is from part of Y
    # test data are used to test our model accuracy 
    # test_size = 0.1 meanning 10% of our data is in test set 
    x_train, x_test, y_train, y_test =  sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    #set linear model to LinearRegression
    model = linear_model.LinearRegression()

    # train our data
    model.fit(x_train, y_train)

    # test model accuracy
    acc =  model.score(x_test, y_test)
   
    if acc > best:
        with open("StudentModel.pickle", "wb") as f:  # open("file name", "wb"  write)
            pickle.dump(model, f)
        best = acc
"""
#print(best) we achieved 94% accuracy
saved_model = open("StudentModel.pickle", "rb")
model = pickle.load(saved_model)

print("C:" , model.coef_)
print("Inter: " , model.intercept_)

#prediction
predictions = model.predict(x_test)

# we stored all the prediction of our model test on test set
# we now pring prediction, test set data, and acctual grade
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "absences"
style.use("ggplot")  #choose gg plot it has grid
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()