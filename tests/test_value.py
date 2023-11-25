import math

import torch
import pytest

from nanograd.value import Value


def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

@pytest.mark.parametrize("data, expected_repr", [(10, 'Value(data=10)'), (-5.5, 'Value(data=-5.5)')])
def test_value_constructor_and_repr(data, expected_repr):
    v = Value(data)
    assert v.data == data
    assert repr(v) == expected_repr

@pytest.mark.parametrize("x, y, result", [(3, 4, 7), (5, -2, 3)])
def test_add(x, y, result):
    a = Value(x)
    b = Value(y)
    c = a + b
    assert c.data == result

@pytest.mark.parametrize("x, y, result", [(3, 4, 12), (5, -2, -10)])
def test_mul(x, y, result):
    a = Value(x)
    b = Value(y)
    c = a * b
    assert c.data == result

@pytest.mark.parametrize("x, power, result", [(2, 3, 8), (5, 2, 25)])
def test_pow(x, power, result):
    a = Value(x)
    b = a ** power
    assert b.data == result

def test_neg():
    a = Value(5)
    b = -a
    assert b.data == -5

@pytest.mark.parametrize("x, y, result", [(10, 3, 7), (5, 5, 0)])
def test_sub(x, y, result):
    a = Value(x)
    b = Value(y)
    c = a - b
    assert c.data == result

@pytest.mark.parametrize("x, y, result", [(3, Value(1), 2), (10, Value(4), 6), (-5, Value(-3), -2)])
def test_rsub(x, y, result):
    c = x - y
    assert c.data == result

@pytest.mark.parametrize("x, y, result", [(10, 2, 5), (9, -3, -3)])
def test_truediv(x, y, result):
    a = Value(x)
    b = Value(y)
    c = a / b
    assert c.data == result

@pytest.mark.parametrize("x", [0, 1, -1])
def test_tanh(x):
    a = Value(x)
    b = a.tanh()
    assert b.data == pytest.approx((math.exp(2*x) - 1) / (math.exp(2*x) + 1))

# Test for exp
@pytest.mark.parametrize("x", [0, 1, -1])
def test_exp(x):
    a = Value(x)
    b = a.exp()
    assert b.data == pytest.approx(math.exp(x))

@pytest.mark.parametrize("x, expected", [(-1, 0), (0, 0), (1, 1)])
def test_relu(x, expected):
    a = Value(x)
    b = a.relu()
    assert b.data == expected

def test_backward():
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + y**2
    z.backward()
    assert x.grad == pytest.approx(3)
    assert y.grad == pytest.approx(8)
