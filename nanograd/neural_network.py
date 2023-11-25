import random
from abc import ABC, abstractmethod

from nanograd.value import Value


class Module(ABC):
    """
    Base class for neural network entitites.

    Provides single interface to reset gradients.
    The same approach is in the PyTorch.
    """

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    @abstractmethod
    def parameters(self) -> list[Value]:
        """Implements listing of parameters for Values."""


class Neuron(Module):
    """Representation of a neuron with random of weights and bias."""

    def __init__(self, nin: int):  # number of inputs
        # weight
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # bias
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list[float | int]) -> Value:
        # w * x + b: raw activation function
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), start=self.b)
        # pass the activation through a non-linearity, or in other words normalize it, so result is -1 <= x <= 1
        out = activation.tanh()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    """Layer of neurons."""

    def __init__(self, nin: int, nout: int):
        # nin - number of inputs
        # nout - number of outputs, how many neurons do we want in the layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[float | int]) -> list[Value] | Value:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        params = []
        for neuron in self.neurons: params.extend(neuron.parameters())
        return params


class MLP(Module):
    "Multilayer Perceptron."

    def __init__(self, nin: int, nouts: list[int]):
        # nin - number of inputs
        # nouts - list of nouts, this list defines sizes of all the layers that we want in the mlp
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list[float | int]) -> list[Value] | Value:
        # run through each layer
        for layer in self.layers:
            out = layer(x)
        # and return the result of the last layer
        return out

    def parameters(self) -> list[Value]:
        params = []
        for layer in self.layers: params.extend(layer.parameters())
        return params
