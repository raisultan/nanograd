import pytest

from nanograd.neural_network import Layer, Neuron, MLP, Module
from nanograd.value import Value


def test_neuron_initialization():
    neuron = Neuron(5)
    assert isinstance(neuron, Neuron)
    assert isinstance(neuron, Module)
    assert len(neuron.w) == 5
    assert all(isinstance(wi, Value) for wi in neuron.w)
    assert isinstance(neuron.b, Value)

@pytest.mark.parametrize("inputs", [[0.5, -0.5, 0.1, -0.2, 0.3], [1, 1, 1, 1, 1]])
def test_neuron_call(inputs):
    neuron = Neuron(len(inputs))
    output = neuron(inputs)
    assert isinstance(output, Value)
    # The output should be between -1 and 1 due to the tanh activation
    assert -1 <= output.data <= 1

def test_neuron_parameters():
    neuron = Neuron(3)
    parameters = neuron.parameters()
    assert len(parameters) == 4  # 3 weights + 1 bias
    assert all(isinstance(param, Value) for param in parameters)

def test_layer_initialization():
    layer = Layer(3, 4)  # 3 inputs, 4 neurons
    assert isinstance(layer, Layer)
    assert isinstance(layer, Module)  # Assuming Module is a base class
    assert len(layer.neurons) == 4
    assert all(isinstance(neuron, Neuron) for neuron in layer.neurons)

def test_layer_call_single_neuron():
    layer = Layer(2, 1)  # 2 inputs, 1 neuron
    inputs = [0.5, -0.5]
    output = layer(inputs)
    assert isinstance(output, Value)
    assert -1 <= output.data <= 1

def test_layer_call_multiple_neurons():
    layer = Layer(2, 3)  # 2 inputs, 3 neurons
    inputs = [0.5, -0.5]
    outputs = layer(inputs)
    assert isinstance(outputs, list)
    assert len(outputs) == 3
    assert all(isinstance(out, Value) for out in outputs)
    assert all(-1 <= out.data <= 1 for out in outputs)

def test_layer_parameters():
    layer = Layer(3, 2)  # 3 inputs, 2 neurons
    parameters = layer.parameters()
    # 2 neurons, each with 3 weights and 1 bias, so 8 parameters in total
    assert len(parameters) == 8
    assert all(isinstance(param, Value) for param in parameters)

def test_mlp_initialization():
    mlp = MLP(3, [4, 2])  # 3 inputs, two layers with 4 and 2 neurons respectively
    assert isinstance(mlp, MLP)
    assert isinstance(mlp, Module)  # Assuming Module is a base class
    assert len(mlp.layers) == 2
    assert all(isinstance(layer, Layer) for layer in mlp.layers)

def test_mlp_call():
    mlp = MLP(2, [3, 1])  # 2 inputs, two layers with 3 and 1 neurons respectively
    inputs = [0.5, -0.5]
    output = mlp(inputs)
    # The final output should be a single Value, as the last layer has only one neuron
    assert isinstance(output, Value)
    assert -1 <= output.data <= 1

def test_mlp_parameters():
    mlp = MLP(3, [2, 2])  # 3 inputs, two layers with 2 neurons each
    parameters = mlp.parameters()
    # First layer: 3 inputs -> 8 parameters (2 neurons, each with 3 weights + 1 bias)
    # Second layer: 2 inputs from first layer -> 6 parameters (2 neurons, each with 2 weights + 1 bias)
    # Total: 8 (first layer) + 6 (second layer) = 14 parameters
    assert len(parameters) == 14

def test_neuron_zero_grad():
    neuron = Neuron(3)  # Neuron with 3 inputs
    for param in neuron.parameters():
        param.grad = 1.0  # Set a non-zero gradient

    neuron.zero_grad()

    for param in neuron.parameters():
        assert param.grad == 0.0

def test_layer_zero_grad():
    layer = Layer(3, 2)  # Layer with 3 inputs and 2 neurons
    for param in layer.parameters():
        param.grad = 1.0  # Set a non-zero gradient

    layer.zero_grad()

    for param in layer.parameters():
        assert param.grad == 0.0

def test_mlp_zero_grad():
    mlp = MLP(3, [2, 2])  # MLP with 3 inputs, two layers with 2 neurons each
    for param in mlp.parameters():
        param.grad = 1.0  # Set a non-zero gradient

    mlp.zero_grad()

    for param in mlp.parameters():
        assert param.grad == 0.0
