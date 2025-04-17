# THUẬT TOÁN LINEAR REGRESSION
import pandas as pd  
import math 
import matplotlib.pyplot as plt 
import numpy as np

data = pd.read_csv('C:\\Learning_IT\\Python\\ML_CORE\\HỌC ML\\DATA\\house_price.csv')
in_train = data['area'].values
out_train = data['price'].values

n = len(in_train)
weight = 0
bias = 0
learning_rate = 0.0001
epochs = 100000
loss_array = []

def loss_function(weight, bias):
	error = 0
	for i in range(n):
		error += ( out_train[i] - (weight * in_train[i] + bias) ) **2
	return error / n

for epoch in range(epochs):
	dw, db = 0, 0
	for i in range(n):
		y_pred = weight * in_train[i] + bias
		error = y_pred - out_train[i]
		dw += 2 * error * in_train[i] 
		db += 2 * error  
	weight -= learning_rate * dw / n 
	bias -= learning_rate * db / n 
	loss_array.append(loss_function(weight, bias))


# plt.scatter(in_train, out_train, color='blue')

x = [i for i in range(30, 150)]
y = [weight * i + bias for i in x]
# plt.plot(x, y, color='red')
x_e = [i for i in range(0, epochs - 1)]
y_e = [loss_array[i] for i in x_e]
plt.plot(x_e, y_e)
plt.show()
