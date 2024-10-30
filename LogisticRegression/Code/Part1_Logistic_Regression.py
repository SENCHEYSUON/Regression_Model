
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000): 
        self.learning_rate = learning_rate 
        self.n_iterations = n_iterations 
        self.weights = None 
        self.bias = None 

    def fit(self, X, y):
        # Number of samples and features
        self.samples, self.features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(self.features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model: wx + b
            linear_model = np.dot(X, self.weights) + self.bias

            # Sigmoid function for logistic regression
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / self.samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / self.samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Return binary predictions
        return [1 if i > 0.5 else 0 for i in y_predicted]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))



