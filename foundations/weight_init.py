import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        weights: List[NDArray[np.float64]],
        biases: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:

        a = x

        for i in range(len(weights)):
            a = np.dot(a, weights[i]) + biases[i]

            # ReLU for hidden layers only
            if i < len(weights) - 1:
                a = np.maximum(0, a)

        return np.round(a, 5)