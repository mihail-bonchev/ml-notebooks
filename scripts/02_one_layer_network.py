import numpy as np

# --- Data: 1 sample, 3 input features ---
X = np.array([[0.6, 0.2, 0.9]])
y_true = np.array([[1.0, 0.0]])  # target outputs for 2 neurons

# --- Initialize weights and biases ---
W = np.random.randn(3, 2)  # 3 inputs -> 2 outputs
b = np.random.randn(1, 2)
lr = 0.1

# --- Forward pass ---
def forward(X, W, b):
    return np.dot(X, W) + b

# --- Loss (MSE) ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --- Training loop ---
for epoch in range(20):
    # Forward
    y_pred = forward(X, W, b)
    loss = mse(y_true, y_pred)

    # --- Gradients ---
    dL_dy = 2 * (y_pred - y_true)           # shape (1, 2)
    dy_dW = X.T                             # shape (3, 1) for each output neuron
    dL_dW = np.dot(X.T, dL_dy)              # shape (3, 2)
    dL_db = dL_dy                           # same shape as bias (1, 2)

    # --- Update weights and biases ---
    W -= lr * dL_dW
    b -= lr * dL_db

    print(f"Epoch {epoch+1:02d}: loss={loss:.4f}")

print("\nFinal predictions:", forward(X, W, b))
