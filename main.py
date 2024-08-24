import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed_forward(self, input):
        total = np.dot(self.weights, input) + self.bias
        return sigmoid(total)

class Neural_Network:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.l1 = Neuron(weights, bias)
        self.l2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feed_forward(self, x):
        out_h1 = self.l1.feed_forward(x)
        out_h2 = self.l2.feed_forward(x)

        return self.o1.feed_forward(np.array([out_h1, out_h2]))

y_pred = np.array([1, 0, 0, 1])
y_true = np.array([0, 0, 0, 0])

print(loss(y_true, y_pred))

weights = np.array([0, 1])
bias = 4

n = Neural_Network()

x = [2, 3]

print(n.feed_forward(x))