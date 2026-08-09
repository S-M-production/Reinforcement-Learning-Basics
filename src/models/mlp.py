import numpy as np
from enum import Enum

#Creating a enum for loss functions
class LossFunction(Enum):
    MSE = 1
    BCE = 2
    CCE = 3

#Creating a enum for weights initialization
class WeightInitialization(Enum):
    Random = 1
    Xavier = 2
    He = 3

class Layer:
    def __init__(self):
        return

    def forward_pass(self,x):
        return

    def backward_pass(self, derivative):
        return

    def update(self, derivative):
        return

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.sigmoid_output = None

    def forward_pass(self,x):
        self.sigmoid_output = 1 / (1 + np.exp(-x))
        return self.sigmoid_output

    def backward_pass(self, derivative):
        return derivative * self.sigmoid_output*(1 - self.sigmoid_output)

#Standard relu as a layer, does not need to know params, will use X to map out gradient then apply it to incoming derivatives
class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.x = None
        self.gradient = None

    def forward_pass(self,x):
        self.gradient = np.where(x > 0, 1, 0)
        self.x = x
        return self.x * self.gradient

    def backward_pass(self, derivative):
        return derivative * self.gradient

#Class for leaky ReLU, does not need to know params, will use X to map out gradient then apply it to incoming derivatives
class LeakyRelu(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.x = None
        self.gradient = None
        self.alpha = alpha
    def forward_pass(self,x):
        self.gradient = np.where(x > 0, 1, self.alpha)
        self.x = x
        return self.x * self.gradient
    def backward_pass(self, derivative):
        return derivative * self.gradient

#Class for softmax layer, does not need to know params, will use X to map out gradient then apply it to incoming derivatives
class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.probabilities = None

    def forward_pass(self,x):
        temp = np.exp(x)
        denominator = np.sum(temp, axis=0, keepdims=True)
        self.probabilities = temp / denominator
        return self.probabilities

    def backward_pass(self, derivative):
        sum = np.sum(derivative * self.probabilities, axis=0, keepdims=True)
        return self.probabilities * (derivative - sum)

    def update(self, derivative):
        return

class Linear(Layer):
    def __init__(self,input_dim,output_dim,learning_rate=0.001,weight_init=WeightInitialization.Random,weights=None,biases=None):
        super().__init__()
        if weights is None:
            match weight_init:
                case WeightInitialization.Random:
                    weights = np.random.uniform(-1,1,size=(output_dim,input_dim))
                case WeightInitialization.Xavier:
                    weights = np.random.uniform(-np.sqrt(6.0/(input_dim + output_dim)),np.sqrt(6.0/(input_dim + output_dim)),size=(output_dim,input_dim))
                case WeightInitialization.He:
                    weights = np.random.normal(0, np.sqrt(2.0/input_dim), size=(output_dim,input_dim))
        if biases is None:
            biases = np.full(shape=(output_dim,1),fill_value=0,dtype=float)
        self.learning_rate = learning_rate
        self.x = None
        self.weights = weights
        self.biases = biases
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward_pass(self,x):
        self.x = x
        return self.weights @ x + self.biases

    def backward_pass(self, derivative):
        return self.weights.T @ derivative

    def update(self, derivative):
        weight_gradient = derivative @ self.x.T
        bias_gradient = np.mean(derivative, axis=1,keepdims=True)

        self.weights -= self.learning_rate * weight_gradient
        self.biases -= self.learning_rate * bias_gradient

class Model:
    def __init__(self, layers):
        self.layers = layers

    def calc_loss(self,y_hat,y,loss_function=LossFunction.MSE):
        match loss_function:
            case LossFunction.MSE:
                return y_hat - y
            case LossFunction.BCE:
                return (y_hat-y)/(y_hat*(1-y_hat))
            case LossFunction.CCE:
                return y*np.log(y_hat)

    def calc_cost(self,y_hat,y,loss_function=LossFunction.MSE):
        match loss_function:
            case LossFunction.MSE:
                return np.mean((y_hat-y)**2)/2
            case LossFunction.BCE:
                return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
            case LossFunction.CCE:
                return -np.mean(np.sum(y*np.log(y_hat),axis=0,keepdims=True))

    def forward_pass(self,x):
        inp = x
        for i in self.layers:
            inp = i.forward_pass(inp)
        return inp

    def backward_pass(self, derivative):
        gradient = derivative
        for i in reversed(self.layers):
            temp_gradient = i.backward_pass(gradient)
            i.update(gradient)
            gradient = temp_gradient

    #Input and output is given in as: batch,input/output
    def train(self,x,y,iterations=10000,cutoff = 0.001,loss_function=LossFunction.MSE):
        cost_history = []
        for i in range(iterations):
            prediction = self.forward_pass(x)
            loss = self.calc_loss(prediction, y, loss_function)
            cost_history.append(self.calc_cost(prediction, y, loss_function))
            self.backward_pass(loss)
            if len(cost_history) < 2:
                continue
            if abs((cost_history[-1]-cost_history[-2])/cost_history[-1]) < cutoff:
                break
        return cost_history

    #Input and output is given in as: batch,input/output
    def advanced_train(self, x, y, epoch, batch_size, iterations=100000, cutoff = 0.001,loss_function=LossFunction.MSE):
        cost_history = []
        counter = 0
        for i in range(epoch):
            batches = self.create_batches(x,y,batch_size)
            for batch in batches:
                batch_x = batch[0]
                batch_y = batch[1]
                prediction = self.forward_pass(batch_x)
                loss = self.calc_loss(prediction, batch_y,loss_function)
                cost_history.append(self.calc_cost(prediction, batch_y,loss_function))
                self.backward_pass(loss)
                if len(cost_history) < 2:
                    continue
                if abs((cost_history[-1]-cost_history[-2])/cost_history[-1]) < cutoff:
                    return cost_history
                if counter == iterations:
                    break
                counter += 1
        return cost_history
        return

    def create_batches(self,x,y,batch_size):
        batches = []
        if batch_size > len(x):
            raise Exception("Batch size cannot be larger than the dataset size.")
        indexes = np.arange(len(x))
        np.random.shuffle(indexes)
        x_shuffled = x[indexes]
        y_shuffled = y[indexes]
        for i in range(0,x.shape[0],batch_size):
            batch_x = x_shuffled[i:i+batch_size].T
            batch_y = y_shuffled[i:i+batch_size].T
            batches.append((batch_x,batch_y))
        return batches