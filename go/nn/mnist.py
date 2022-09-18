import os
from typing import Tuple, List

import numpy as np
from numpy import typing as nptype


def encode_label(j):  # <1>
    """one-hot encode indices to vectors of length 10"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# A list of tuples, where the first is a 784-element ndarray of uint8, the
# second a 10-element ndarray of float64
Feature = Tuple[nptype.NDArray[np.uint8], nptype.NDArray[np.float64]]
FeatureList = List[Feature]

def shape_data(data: np.ndarray) -> FeatureList:
    """shape data into a list of tuples of (image, labels)

    the image is a 784-element ndarray vector of uint8
    the labels are 10-element ndarray vector of float64"""
    # Flatten the input images to feature vectors of length 784.
    features = [np.reshape(x, (784, 1)) for x in data[0]]

    # All labels are one-hot encoded.
    labels = [encode_label(y) for y in data[1]]

    # Create pairs of features and labels.
    return list(zip(features, labels))


# I've not bothered to type the return here. A man has limits.
def load_data_impl():
    # file retrieved by:
    #   wget https://s3.amazonaws.com/img-datasets/mnist.npz -O code/dlgo/nn/mnist.npz
    # code based on:
    #   site-packages/keras/datasets/mnist.py
    path = 'mnist.npz'
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, path)
    f = np.load(abs_file_path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_data() -> Tuple[FeatureList, FeatureList]:
    train_data, test_data = load_data_impl()
    return shape_data(train_data), shape_data(test_data)
