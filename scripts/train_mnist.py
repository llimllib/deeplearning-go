import sys
import os
sys.path.insert(0, os.path.dirname(sys.argv[0]) + "/..")

from go.nn.layer import SequentialNetwork, DenseLayer, ActivationLayer
from go.nn.mnist import load_data


def main():
    training_data, test_data = load_data()
    net = SequentialNetwork()
    net.add(DenseLayer(784, 392))
    net.add(ActivationLayer(392))
    net.add(DenseLayer(392, 196))
    net.add(ActivationLayer(196))
    net.add(DenseLayer(196, 10))
    net.add(ActivationLayer(10))

    net.train(
        training_data,
        epochs=10,
        mini_batch_size=10,
        learning_rate=3.0,
        test_data=test_data,
    )

# TODO: move this to scripts/
if __name__ == "__main__":
    main()
