import unittest

from yane.util.activation import ActivationFunction, ActivationType


class TestActivationFunction(unittest.TestCase):

    def test_linear(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.LINEAR, 1.0), 1.0)
        self.assertEqual(ActivationFunction.activate(ActivationType.LINEAR, -5.0), -5.0)

    def test_sigmoid(self):
        result = ActivationFunction.activate(ActivationType.SIGMOID, 0.0)
        self.assertAlmostEqual(result, 0.5)

    def test_sigmoid_large_positive(self):
        result = ActivationFunction.activate(ActivationType.SIGMOID, 1000.0)
        self.assertAlmostEqual(result, 1.0)

    def test_tanh(self):
        self.assertAlmostEqual(ActivationFunction.activate(ActivationType.TANH, 0.0), 0.0)

    def test_relu_negative(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.RELU, -1.0), 0.0)

    def test_relu_positive(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.RELU, 2.0), 2.0)

    def test_binary_above_threshold(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.BINARY, 1.0), 1.0)

    def test_binary_below_threshold(self):
        self.assertEqual(ActivationFunction.activate(ActivationType.BINARY, 0.0), 0.0)

    def test_unknown_raises(self):
        with self.assertRaises((ValueError, Exception)):
            ActivationFunction.activate("nonexistent", 0.0)


if __name__ == "__main__":
    unittest.main()
