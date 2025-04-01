X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Initial weights as given
initial_weights = np.array([10, 0.2, -0.75])

# Train with step activation
trained_weights, errors, epochs_needed = train_perceptron(X_and, y_and, initial_weights.copy())

print(f"Epochs needed for convergence with step activation: {epochs_needed}")

