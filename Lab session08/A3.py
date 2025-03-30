import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

activation_functions = ['step', 'bipolar_step', 'sigmoid', 'relu']
epochs_results = {}

for activation in activation_functions:
    _, _, epochs = train_perceptron(X_and, y_and, initial_weights.copy(), activation=activation)
    epochs_results[activation] = epochs
    print(f"Epochs needed with {activation} activation: {epochs}")
