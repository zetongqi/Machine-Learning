import numpy as np
from sklearn import preprocessing
from arff_parser import *


# return X and Y as numpy array
def feature_scaling(raw):
	X = []
	Y = []
	for row in raw:
		X.append(row[:-1])
		Y.append(row[-1])
	X = np.asarray(X)
	X = preprocessing.scale(X)
	Y = np.asarray(Y)
	return X, Y


def change_label_encoding(Y, label_list):
	for i in range(Y.shape[0]):
		for j in range(len(label_list)):
			if Y[i] == label_list[j]:
				Y[i] = j
	return Y


def process_data(raw, label_list):
	X, Y = feature_scaling(raw)
	Y = change_label_encoding(Y, label_list)
	Y = Y.astype(int)
	return X, Y


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def generate_batch(X, Y, batchsize):
	for i in np.arange(0, len(X), batchsize):
		yield X[i:i+batchsize], Y[i:i+batchsize]


# h and y are vectors
def cross_entropy(h, y):
	# return the average
	return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def forward_propagation(x, theta1, theta2):
	z2 = np.dot(x, theta1)
	a2 = sigmoid(z2)
	z3 = np.dot(a2, theta2)
	a3 = sigmoid(z3)
	return a2, a3


def batch_backward_propagation(x_batch, y_batch, theta1, theta2):
	D1 = np.zeros(theta1.shape)
	D2 = np.zeros(theta2.shape)
	for i in range(x_batch.shape[0]):
		a2, a3 = forward_propagation(x_batch[i], theta1, theta2)
		delta3 = a3 - y_batch[i]
		delta2 = np.multiply(np.multiply(np.dot(theta2, delta3), a2.T), (1-a2).T)
		D2 = np.add(D2, np.dot(delta3, a2))
		D1 = np.add(D1, np.outer(delta2, x_batch[i]).T)
	return D1, D2


def initial_theta(layer1_num, layer2_num):
	theta1 = np.random.uniform(-0.01, 0.01, [layer1_num, layer2_num])
	theta2 = np.random.uniform(-0.01, 0.01, layer2_num)
	return theta1, theta2


def get_test_accuracy(X_test, Y_test, theta1, theta2):
	cnt = 0
	for i in range(X_test.shape[0]):
		_, h = forward_propagation(X_test[i], theta1, theta2)
		if h >= 0.5:
			h = 1
		else:
			h = 0
		if h == Y_test[i]:
			cnt += 1
	return cnt/X_test.shape[0]


def neural_network_training(train_arff, test_arff, e, l, h):
	X, Y = process_data(train_arff.data, train_arff.label.attribute_list)
	X_test, Y_test = process_data(test_arff.data, test_arff.label.attribute_list)
	theta1, theta2 = initial_theta(len(X[0]), 10)
	for e in range(e):
		for x_batch, y_batch in generate_batch(X, Y, 100):
			D1, D2 = batch_backward_propagation(x_batch, y_batch, theta1, theta2)
			theta1 = theta1 - l * D1
			theta2 = theta2 - l * D2
		print("epoch: ", e, sep="")
		H = []
		for i in range(X_test.shape[0]):
			_, h = forward_propagation(X_test[i], theta1, theta2)
			H.append(h)
		H = np.asarray(H)
		print("error: ", cross_entropy(H, Y_test), sep="")
		print("accuracy: ", get_test_accuracy(X_test, Y_test, theta1, theta2))


d = arff_data("diabetes_train.arff")
a = arff_data("diabetes_test.arff")
neural_network_training(d, a, 100, 0.01, 10)
