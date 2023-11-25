from __future__ import annotations
import math
from typing import Callable


class Value:
    """Representation of singular scalar value with its gradient."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = '', label: str = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

        self.label = label
        # derivative of the output with respect to current value, yet no effect
        self.grad = 0.0

    def __repr__(self) -> int:
        return f'Value(data={self.data})'

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward() -> None: self.grad += 1 * out.grad; other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward() -> None: self.grad += out.grad * other.data; other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other: Value) -> Value:
        return self * other

    def __pow__(self, other: Value | int) -> Value:
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward() -> None: self.grad += other * self.data**(other-1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value) -> Value:
        return other + (-self)

    def __truediv__(self, other: Value) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: Value) -> Value:
        return other * self**-1

    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self, ), _op='tanh')

        def _backward() -> None: self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        exp_x = math.exp(self.data)
        out = Value(exp_x, (self,), 'exp')

        def _backward() -> None: self.grad += exp_x * out.grad
        out._backward = _backward

        return out

    def relu(self) -> Value:
        """ReLU(x)=max(0,x)."""
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward() -> None: self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Runs backward pass."""
        topo = []
        visited = set()

        def _build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    _build_topo(child)
                topo.append(node)

        _build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
