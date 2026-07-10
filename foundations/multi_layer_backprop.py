import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert to numpy arrays
        x = np.array(x, dtype=np.float64)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)

        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)

        y_true = np.array(y_true, dtype=np.float64)

        # ===== Forward Pass =====
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)      # ReLU

        z2 = W2 @ a1 + b2
        y_pred = z2

        loss = np.mean((y_pred - y_true) ** 2)

        # ===== Backward Pass =====

        n = y_true.size

        # dL/dy_pred
        dL_dy = (2 * (y_pred - y_true)) / n

        # Gradients for W2 and b2
        dW2 = np.outer(dL_dy, a1)
        db2 = dL_dy

        # Backprop through second linear layer
        da1 = W2.T @ dL_dy

        # Backprop through ReLU
        dz1 = da1 * (z1 > 0)

        # Gradients for W1 and b1
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }
