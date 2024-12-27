from Value import Value
import random

class Neuron:

    def __init__(self, input_size):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        total = self.bias
        for i in range(len(self.weights)):
            total += x[i] * self.weights[i]
        return total.tanh()

    def parameters(self):
        return [self.bias] + self.weights

class Layer:

    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, input_layer):
        results = []
        for i in range(len(self.neurons)):
            results.append(self.neurons[i](input_layer))
        return results

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:

    def __init__(self, input_size, layers_size):
        self.input_size = input_size
        self.layers_size = layers_size

        list_layers = [self.input_size] + self.layers_size
        self.layers = [Layer(list_layers[i], list_layers[i+1]) for i in range(len(list_layers)-1)]

    def __call__(self, x):
        result = x
        for i in range(len(self.layers)):
            result = self.layers[i](result)
        return result

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def train_step(self, X, y, learning_rate=0.01):
        ypred = self(X)
        loss = sum((yout - ygt)**2 for yout, ygt in zip(ypred, y))
        loss.backward()

        for p in self.parameters():
            p.data -= learning_rate * p.grad
            p.grad = 0
        
        return loss.data