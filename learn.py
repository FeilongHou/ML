import sklearn
import mglearn
from sklearn import linear_model

x, y = mglearn.datasets.load_extended_boston()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.2, random_state = 0)
# random_state makes sure that it splits the dataset the same way every time we run

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

print(model.coef_)  # theta
print(model.intercept_) # theta_0

model.score(x_test, y_test)

# Ridge regression
ridge = sklearn.linear_model.Ridge().fit(x_train, y_train)

print(ridge.score(x_test, y_test))

# Ridge with alpha = 10
ridge = sklearn.linear_model.Ridge(alpha = 10).fit(x_train, y_train)
print(ridge.score(x_test, y_test))

# Lasso
las = sklearn.linear_model.Lasso()
las.fit(x_train, y_train)
print(las.score(x_test, y_test))