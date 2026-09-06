import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # Convert to numpy arrays for easier computation
        x = np.array(x, dtype=np.float64)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)
        
        # --- Forward Pass ---
        # Layer 1: Linear
        z1 = np.dot(W1, x) + b1  # shape: (hidden_size,)
        
        # ReLU activation
        a1 = np.maximum(0, z1)   # shape: (hidden_size,)
        
        # Layer 2: Linear (output layer)
        z2 = np.dot(W2, a1) + b2  # shape: (output_size,)
        predictions = z2  # No activation (linear output)
        
        # --- Loss: Mean Squared Error ---
        # MSE = (1/n) * sum((pred - y_true)^2)
        n = len(y_true)
        loss = np.mean((predictions - y_true) ** 2)
        
        # --- Backward Pass (Backpropagation) ---
        # Gradient of loss w.r.t. predictions: dL/dpred = (2/n) * (pred - y_true)
        dL_dpred = (2 / n) * (predictions - y_true)  # shape: (output_size,)
        
        # --- Gradients for Layer 2 ---
        # dL/dW2 = dL/dpred * (dpred/dz2) * (dz2/dW2)
        # dpred/dz2 = 1 (since predictions = z2, no activation)
        # dz2/dW2 = a1^T (transpose of a1)
        dW2 = np.outer(dL_dpred, a1)  # shape: (output_size, hidden_size)
        
        # dL/db2 = dL/dpred * 1
        db2 = dL_dpred.copy()  # shape: (output_size,)
        
        # --- Gradient for Layer 1 (through ReLU) ---
        # Gradient w.r.t. a1: dL/da1 = W2^T * dL/dpred
        dL_da1 = np.dot(W2.T, dL_dpred)  # shape: (hidden_size,)
        
        # Gradient through ReLU: da1/dz1 = 1 if z1 > 0 else 0
        # dL/dz1 = dL/da1 * (z1 > 0)
        dL_dz1 = dL_da1 * (z1 > 0)  # shape: (hidden_size,)
        
        # --- Gradients for Layer 1 ---
        # dL/dW1 = dL/dz1 * (dz1/dW1)
        # dz1/dW1 = x^T
        dW1 = np.outer(dL_dz1, x)  # shape: (hidden_size, input_size)
        
        # dL/db1 = dL/dz1 * 1
        db1 = dL_dz1.copy()  # shape: (hidden_size,)
        
        # --- Round all values to 4 decimals ---
        loss = np.round(loss, 4)
        dW1 = np.round(dW1, 4).tolist()
        db1 = np.round(db1, 4).tolist()
        dW2 = np.round(dW2, 4).tolist()
        db2 = np.round(db2, 4).tolist()
        
        return {
            'loss': float(loss),
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }