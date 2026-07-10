import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Formula: 1 / (1 + e^(-z))
        return np.round(1 / (1 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Formula: max(0, z) element-wise
        return np.maximum(0, z).astype(np.float64)
