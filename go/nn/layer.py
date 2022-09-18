from abc import ABC, abstractmethod
from typing import Optional
import random

import numpy as np
from numpy import typing as nptype

from go.nn.mnist import Feature, FeatureList


class Layer:
    def __init__(self):
        self.params = []

        # A layer knows its predecessor
        self.previous: Optional[Layer] = None
        self.next: Optional[Layer] = None

        # Each layer can persist data flowing into and out of it in the forward
        # pass
        self.input_data = None
        self.output_data = None

        # Analogously, a layer holds input and output data for the backward
        # pass
        self.input_delta = None
        self.output_delta = None

    def connect(self, layer: "Layer"):
        self.previous = layer
        layer.next = self

    # Each layer implementation has to provide a function to feed data forward
    def forward(self):
        raise NotImplementedError

    # input_data is reserved for the first layer; all others get their input
    # from the previous output
    def get_forward_input(self) -> nptype.NDArray[np.float64]:
        if self.previous:
            return self.previous.output_data
        else:
            return self.input_data

    # Layers have to implement backprop of error terms - a way to feed input
    # errors backward through the network
    def backward(self):
        raise NotImplementedError

    # input delta is reserved for the last layer; all other layers get their
    # error terms from their successor
    def get_backward_input(self) -> nptype.NDArray[np.float64]:
        if self.next:
            return self.next.output_delta
        else:
            return self.input_delta

    # you compute and accumulate deltas per mini-batch, after which you need to
    # reset these deltas
    def clear_deltas(self):
        pass

    # update layer parameters according to current deltas, using the specified
    # learning_rate
    def update_params(self, learning_rate):
        # rn the typechecker doesn't like that the learning_rate isn't accessed
        # - why can't we raise a NotImplementedError?
        pass

    def describe(self):
        raise NotImplementedError


# so the type here is weird: sigmoid_scalar will happily work on any
# nptype.ArrayLike, but the authors intend it to be a scalar-only version. I've
# given it float -> float, but it will return an ArrayLike if an ArrayLike is
# input. Not sure if there is a type that includes float and np.float64, for
# example? Maybe I'm being precious and this should go untyped. Unclear.
#
# I changed the name from sigmoid_double to sigmoid_scalar because "double"
# doesn't make sense in python
def sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z: nptype.ArrayLike) -> nptype.ArrayLike:
    return np.vectorize(sigmoid_scalar)(z)


# same type issues here as above
def sigmoid_prime_scalar(x: float) -> float:
    """the derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_prime(z: nptype.ArrayLike) -> nptype.ArrayLike:
    return np.vectorize(sigmoid_prime_scalar)(z)


# an activation layer using the sigmoid function to activate neurons
class ActivationLayer(Layer):
    def __init__(self, input_dim: int):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        data = self.get_forward_input()
        # The forward pass is simply applying the sigmoid to the input data
        self.output_data = sigmoid(data)

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        # The backward pass is element-wise multiplication of the error term
        # with the sigmoid derivative evaluated at the input to this layer
        self.output_delta = delta * sigmoid_prime(data)

    def describe(self):
        print(f"|-- {self.__class__.__name__}")
        print(f"  |-- dimensions: ({self.input_dim}, {self.output_dim})")


class DenseLayer(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Randomly initialize weight matrix and bias vector. There are many
        # more sophisticated ways to initialize parameters so that they more
        # accurately reflect the structure of your data, but this is an
        # acceptable baseline. More on this in chapter 6
        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

        # the layer parameters consist of weights and bias terms
        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        data = self.get_forward_input()
        # The forward pass of the dense layer is the affine-linear
        # transformation on input data defined by weights and biases
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        """

        To compute the delta for this layer, you need to transpose W and
        multiply it by the incoming delta WtΔ. The gradients for W an b are
        also easily computed: ΔW = Δyt and Δb = Δ, where y denotes the input to
        this layer
        """
        data = self.get_forward_input()
        delta = self.get_backward_input()

        # The current delta is added to the bias delta
        self.delta_b += delta
        # Then you add this term to the weight delta
        self.delta_w += np.dot(delta, data.transpose())
        self.output_delta = np.dot(self.weight.transpose(), delta)

    # the update rule for this layer is given by accumulating the deltas,
    # according to the learning rate you specify for your network
    def update_params(self, rate: float):
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):
        print(f"|-- {self.__class__.__name__}")
        print(f"  |-- dimensions: ({self.input_dim}, {self.output_dim})")


class ErrorType(ABC):
    @staticmethod
    @abstractmethod
    def loss_function(
        predictions: nptype.ArrayLike, labels: nptype.ArrayLike
    ) -> nptype.ArrayLike:
        ...

    @staticmethod
    @abstractmethod
    def loss_derivative(
        predictions: nptype.ArrayLike, labels: nptype.ArrayLike
    ) -> nptype.ArrayLike:
        ...


class MSE:
    def __init__(self):
        pass

    @staticmethod
    def loss_function(predictions: nptype.ArrayLike, labels: nptype.ArrayLike):
        diff = predictions - labels
        # we have to ignore the typing here because pyright doesn't believe
        # that you can sum two matrices and end up with a matrix. Ex:
        #
        # In [14]: sum(np.arange(20).reshape(4,5))
        # Out[14]: array([30, 34, 38, 42, 46])
        return 0.5 * sum(diff * diff)[0]  # type: ignore

    @staticmethod
    def loss_derivative(
        predictions: nptype, labels: nptype.ArrayLike
    ) -> nptype.ArrayLike:
        return predictions - labels


class SequentialNetwork:
    def __init__(self, loss: Optional[ErrorType] = None):
        print("Initializing network")
        self.layers = []
        if loss is None:
            self.loss = MSE()

    def add(self, layer: Layer):
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(
        self,
        training_data: FeatureList,
        epochs: int,
        mini_batch_size,
        learning_rate: float,
        test_data: Optional[FeatureList] = None,
    ):
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, learning_rate)
            if test_data:
                n_test = len(test_data)
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch} completed")

    def train_batch(self, mini_batch: FeatureList, learning_rate: float):
        self.forward_backward(mini_batch)
        self.update(mini_batch, learning_rate)

    def update(self, mini_batch: FeatureList, learning_rate: float):
        # a common technique is to normalize the learning rate by the
        # mini-batch size
        learning_rate = learning_rate / len(mini_batch)

        for layer in self.layers:
            layer.update_params(learning_rate)
        for layer in self.layers:
            layer.clear_deltas()

    def forward_backward(self, mini_batch: FeatureList):
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:
                layer.forward()
            self.layers[-1].input_delta = self.loss.loss_derivative(
                self.layers[-1].output_data, y
            )
            for layer in reversed(self.layers):
                layer.backward()

    def single_forward(self, x: Feature) -> np.ndarray:
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data: FeatureList) -> int:
        test_results = [
            (np.argmax(self.single_forward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)
