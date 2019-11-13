import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#1
data = pd.read_csv('ex1data1.csv')

#2
# data.plot(y='profit', x='population', kind='scatter')
# plt.show()

#3
x = np.array(data.values[:, :data.shape[1] - 1])
y = np.array(data.values[:, data.shape[1] - 1])

m_x = x.mean(axis=0)
std_x = x.std(axis=0)
x = (x - m_x)/std_x

x = np.hstack((x, np.ones(data.shape[0]).reshape(data.shape[0], 1)))

#4
def mserror(y, y_predict):
    return np.mean((y - y_predict)**2)

def lin_prediction(X, w):
    return np.dot(X, w)

# def w_matrix(X, Y):
#     return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
#
#
# w = w_matrix(x, y)
# y_pr = lin_prediction(x, w)
#
# er = mserror(y, y_pr)
# print('W: {}. Linear prediction error (matrix form): {}'.format(w, er))

#5
def gradient(X, Y, m, w, alpha):
    return alpha / m * np.dot(X.T, (Y - lin_prediction(X, w)))

#Full batch gradient descent
def full_batch_gradient_descent(x, y, w_init, n=2000, alpha=0.01):
    w = w_init
    mse = []
    for i in range(n):
        w += gradient(x, y, x.shape[0], w, alpha)
        mse.append(mserror(y, lin_prediction(x, w)))
    return w, mse

#Mini batch gradient descent
def mini_batch_gradient_descent(x, y, w_init, batch_size=32, batch_num=5000, n=1, alpha=0.001):
    w = w_init
    mse = []
    for j in range(batch_num):
        mini_batch_index = np.random.randint(0, x.shape[0], batch_size)
        x_mini_batch = np.array([x[i] for i in mini_batch_index])
        y_mini_batch = np.array([y[i] for i in mini_batch_index])
        for i in range(n):
            w += gradient(x_mini_batch, y_mini_batch, batch_size, w, alpha)
            mse.append(mserror(y, lin_prediction(x, w)))
    return w, mse

#Stochastic gradient descent
def stochastic_gradient_descent(x, y, w_init, n=5000, alpha=0.001):
    w = w_init
    mse = []
    for j in range(n):
        rand_index = np.random.randint(0, x.shape[0])
        w += gradient(x[rand_index], y[rand_index], x.shape[0], w, alpha)
        mse.append(mserror(y, lin_prediction(x, w)))
    return w, mse


# #6
# mini_batch_gradient_descent(x, y, [4., 5.])

# reg = LinearRegression().fit(x.T[0].reshape(-1, 1), y)
# print('w1:', reg.coef_, 'w0', reg.intercept_, 'er', mserror(y, reg.predict(x.T[0].reshape(-1, 1))))
