#!/usr/bin/env python3
# Single Perceptron Algorithm on the Sonar Dataset
# Classify whether the signal bounce off metal cylinder(mine) or rock
import random
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float(in the input)
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Converting the class strings to integers(0/1)
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for value,key in enumerate(unique):
		lookup[key] = value
	for row in dataset:
		row[column] = lookup[row[column]]

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

def evaluate_algorithm(dataset,l_rate,n_epoch):
	scores = list()
	train_set=list(dataset)
	test_set=list()
	actual=list()
	#testing on 50 examples
	for i in range(200):
		index=random.randint(1,207)
		test_set.append(dataset[index])
		actual.append(dataset[index][-1])

	predicted = perceptron(train_set, test_set,l_rate,n_epoch)
	accuracy = accuracy_metric(actual, predicted)
	scores.append(accuracy)
	return scores

# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)

# evaluate algorithm
l_rate = 0.5
n_epoch = 1000
scores = evaluate_algorithm(dataset,l_rate, n_epoch)
print('Score: %s' % scores)