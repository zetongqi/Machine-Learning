from arff_parser import *
import random
import math
import numpy as np
from sklearn import preprocessing


# scale each feature of feature data X, and convert label to numeric coding given a label_list
def feature_scaling(raw, label_list):
	X = []
	Y = []
	for row in raw:
		X.append(row[:-1])
		Y.append(row[-1])
	X = np.asarray(X)
	X = preprocessing.scale(X)
	X = X.tolist()
	for i in range(len(X)):
		X[i] = [1] + X[i]
	for i in range(len(Y)):
		for j in range(len(label_list)):
			if Y[i] == label_list[j]:
				Y[i] = j
	return X, Y


def generate_batch(X, Y, batchsize):
	for i in np.arange(0, len(X), batchsize):
		yield X[i:i+batchsize], Y[i:i+batchsize]


# return initialized theta between -0.01 and 0.01
def initialize_theta(num_of_features):
	theta = []
	for i in range(num_of_features):
		theta.append(random.uniform(-0.01, 0.01))
	return theta


def sigmoid(z):
	return 1 / (1 + math.exp(-z))


def sum_of_multiplication(v1, v2):
	if len(v1) != len(v2):
		return None
	else:
		mul = 0
		for i in range(len(v1)):
			mul += v1[i] * v2[i]
		return mul


def cross_entropy(theta, x_batch, y_batch):
	CrossEntropy = 0
	for i in range(len(x_batch)):
		mul = sum_of_multiplication(theta, x_batch[i])
		o = sigmoid(mul)
		y = y_batch[i]
		CrossEntropy += -y * math.log(o) - (1-y) * math.log(1 - o)
	CrossEntropy = CrossEntropy/len(x_batch)
	return CrossEntropy


def cross_entropy_derivative(theta, x_batch, y_batch):
	gradients = [0] * len(theta)
	for i in range(len(gradients)):
		for j in range(len(x_batch)):
			mul = sum_of_multiplication(theta, x_batch[j])
			o = sigmoid(mul)
			y = y_batch[j]
			gradients[i] += o - y
	for i in range(len(gradients)):
		gradients[i] = gradients[i] / len(x_batch)
	return gradients


def update_theta(theta, gradients, l):
	if len(theta) != len(gradients):
		return None
	else:
		for i in range(len(theta)):
			theta[i] += l * gradients[i]
	return theta


def stochastic_gradient_Descent(raw, label_list, e, l, batch_size):
	theta = initialize_theta(len(raw[0]))
	for epoch in range(e):
		random.shuffle(raw)
		X, Y = feature_scaling(raw, label_list)
		error = 0
		for x_batch, y_batch in generate_batch(X, Y, batch_size):
			error += cross_entropy(theta, x_batch, y_batch)
			gradients = cross_entropy_derivative(theta, x_batch, y_batch)
			theta = update_theta(theta, gradients, l)
		print(error)
	return theta


def classify(row, theta):
	o = sum_of_multiplication(row, theta)
	if o >= 0.5:
		return 1
	else:
		return 0


def get_test_accuracy(test_raw, theta, label_list):
	X_test, Y_test = feature_scaling(test_raw, label_list)
	cnt = 0
	for i in range(len(X_test)):
		if classify(X_test[i], theta) == Y_test[i]:
			cnt += 1
	print("Accuracy: ", cnt / len(test_raw) * 100, "%", sep = "")


theta = stochastic_gradient_Descent(arff_data("magic_train.arff").data, ["g", "h"], 11, 0.0001, 20)

get_test_accuracy(arff_data("magic_test.arff").data, theta, ["g", "h"])

