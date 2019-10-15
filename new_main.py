# cnn model
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from my_models import CNN_1D
import torch
import torch.nn as nn
import sys
from matplotlib import pyplot
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# from keras.utils import to_categorical

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.to_numpy()

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements

def to_categorical(y, num_classes=6, dtype='float32'):
	y = np.array(y, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype=dtype)
	categorical[np.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = np.reshape(categorical, output_shape)
	return categorical

def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	# print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	# print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	# trainy = to_categorical(trainy)
	# testy = to_categorical(testy)

	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))

	print(str(" Train: "), trainX.shape, trainy.shape, str(" Test: "), testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
# def evaluate_model(trainX, trainy, testX, testy):
# 	verbose, epochs, batch_size = 0, 10, 32
# 	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
# 	model = Sequential()
# 	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
# 	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# 	model.add(Dropout(0.5))
# 	model.add(MaxPooling1D(pool_size=2))
# 	model.add(Flatten())
# 	model.add(Dense(100, activation='relu'))
# 	model.add(Dense(n_outputs, activation='softmax'))
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# fit network
# 	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
# 	# evaluate model
# 	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
# 	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# Hyperparameters
num_epochs = 10
num_classes = 6
batch_size = 32
learning_rate = 0.001


# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	n_timesteps, n_features, n_outputs = trainX.shape[2], trainX.shape[1], trainy.shape[1]

	model = CNN_1D()
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


	total_training_length = trainX.shape[0]

	# total_training_length = batch_size * 10
	num_epochs = 10
	loss_list = []
	acc_list = []

	# sys.exit()
	print("total_training_length=> ", str(total_training_length))
	for epoch in range(num_epochs):
		correct_all = 0
		total_all = 0

		for i in range(0, total_training_length, batch_size):
			if total_training_length - i < batch_size:
				break

			train_x_batch = trainX[i:i + batch_size]
			train_y_batch = trainy[i:i + batch_size]

			train_x_batch = torch.from_numpy(train_x_batch)
			train_y_batch = torch.from_numpy(train_y_batch).view(-1)

			# print("train_y_batch ", train_y_batch.shape)
			# return
			# Run the forward pass
			# print("batch=> ", str(i/batch_size))
			outputs = model(train_x_batch.float())
			# print("outputs ", outputs.shape)
			# print(train_y_batch.view(-1))
			loss = criterion(outputs, train_y_batch)

			loss_list.append(loss.item())

			# # Backprop and perform Adam optimisation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Track the accuracy
			total = train_y_batch.size(0)
			_, predicted = torch.max(outputs.data, 1)
			correct = (predicted == train_y_batch).sum().item()
			acc_list.append(correct / total)

			correct_all += correct
			total_all += total

			# if (i == batch_size) or False:
			# 	# print("train_y_batch", str(train_y_batch))
			# 	# print("predicted", str(predicted))
			# 	# print("....")
			# 	# print("correct", str(correct))
			# 	# print("total size", str(total))
			#
			# 	print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
			# 		  .format(epoch + 1, num_epochs, i, total_training_length, loss.item(),
			# 				  (correct / total) * 100))
		# accuracy of each epoch
		print('Epoch [{}/{}]'.format(epoch + 1, num_epochs), ", Accuracy: ", str((correct_all / total_all) * 100))

	# Test the model
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		total_test_length = testX.shape[0]

		for i in range(0, total_test_length, batch_size):
			if total_test_length - i < batch_size:
				break

			test_x_batch = testX[i:i + batch_size]
			test_y_batch = testy[i:i + batch_size]

			test_x_batch = torch.from_numpy(test_x_batch)
			test_y_batch = torch.from_numpy(test_y_batch).view(-1)

			outputs = model(test_x_batch.float())
			_, predicted = torch.max(outputs.data, 1)
			total += test_y_batch.size(0)
			correct += (predicted == test_y_batch).sum().item()

		print('Test Accuracy of the model on the test data is: {} %'.format((correct / total) * 100))


	# # repeat experiment
	# scores = list()
	# for r in range(repeats):
	# 	score = evaluate_model(trainX, trainy, testX, testy)
	# 	score = score * 100.0
	# 	print('>#%d: %.3f' % (r+1, score))
	# 	scores.append(score)
	# # summarize results
	# summarize_results(scores)

# run the experiment
run_experiment()


# a = torch.randn(32, 101, 9)  # [batch_size, in_channels, len]
# m = nn.Conv1d(100, 100, 2)   # in_channels, out_channels, kernel_size
# out = m(a)
# print(out.size())
# print(m)

# a = np.random.randint(2, size=(32, 128, 9))
# print(len(a[0]))
# a = np.reshape(a, (32, 9, 128))
# print(a.shape)
# print(len(a[0]))

# dd = load_file("HARDataset/train/Inertial Signals/body_acc_x_train.txt")
# print(dd.size)

# The model requires a three-dimensional input with [samples, time steps, features]
# The output for the model will be a six-element vector containing the probability of each class [0->5] (len = 6)

# Train (7352, 128, 9) (7352, 6)
# Test  (2947, 128, 9) (2947, 6)







