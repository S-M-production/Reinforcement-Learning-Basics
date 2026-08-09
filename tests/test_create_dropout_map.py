import unittest

import pytest

import models.mlp as mlp
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_dropout_percent(self):
        dropout = 0.3
        model = mlp.Model([mlp.Linear(input_dim=1,output_dim=10,dropout=dropout)])
        model.layers[0].create_dropout_map()
        result = np.average(model.layers[0].dropout_map)
        assert np.average(model.layers[0].dropout_map) == 1-dropout
    def test_dropout_randomness(self):
        dropout = 0.3
        model = mlp.Model([mlp.Linear(input_dim=1,output_dim=10,dropout=dropout)])
        model.layers[0].create_dropout_map()
        result1 = model.layers[0].dropout_map
        model.layers[0].create_dropout_map()
        result2 = model.layers[0].dropout_map
        assert not np.array_equal(result1, result2)

if __name__ == '__main__':
    unittest.main()
