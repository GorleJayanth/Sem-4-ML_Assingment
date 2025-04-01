def train_perceptron(X, y, weights, learning_rate=0.05, activation='step', max_epochs=1000, convergence_error=0.002):
    epoch_errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Forward pass
            summation = summation_unit(xi, weights)
            output = activation_unit(summation, activation)
            # Error calculation
            error = comparator_unit(target, output)
            total_error += error ** 2
            # Weight updates
            weights[0] += learning_rate * error  # bias update
            for i in range(len(xi)):
                weights[i+1] += learning_rate * error * xi[i]
        epoch_errors.append(total_error)
