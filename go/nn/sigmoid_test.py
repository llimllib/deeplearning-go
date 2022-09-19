import time

import numpy as np
from scipy.special import expit
from go.nn.layer import ActivationLayer

from .layer import sigmoid, sigmoid_prime, sigmoid_prime_scipy


def test_sigmoid():
    matrix = np.arange(4).reshape(2, 2)
    sig = sigmoid(matrix)
    expected = np.array([[0.5, 0.73105858], [0.88079708, 0.95257413]])
    assert np.allclose(sig, expected), f"{sig} != {expected}"


def test_equiv():
    for _ in range(100):
        matrix = np.random.rand(10, 10)
        sig = sigmoid(matrix)
        scip = expit(matrix)
        assert np.allclose(sig, scip), f"{sig} != {scip}"


def test_equiv_deriv():
    for _ in range(100):
        matrix = np.random.rand(10, 10)
        sig = sigmoid_prime(matrix)
        scip = sigmoid_prime_scipy(matrix)
        assert np.allclose(sig, scip), f"{sig} != {scip}"


# TODO: move this into a benchmarking script
# def test_benchmark_ActivationLayer():
#     nsteps = 100
#     npoints = 100_000
#     training_data = [np.random.rand(10, 1) for _ in range(npoints)]
#     delta = np.random.rand(10, 1)
#
#     t1 = time.time()
#     layer = ActivationLayer(10)
#     for i in range(nsteps):
#         layer.input_data = training_data
#         layer.forward()
#         layer.input_delta = delta
#         layer.backward()
#     t2 = time.time()
#     print(f"{nsteps} steps with {npoints} points in {t2 - t1} seconds")
