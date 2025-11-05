import numpy as np

# Input and target
x = np.array([0.5])
y_true = np.array([1.0])

# Random initial weights and bias
w = np.random.randn(1)
b = np.random.randn(1)

# Learning rate
lr = 0.1

# --- Forward pass ---
def forward(x, w, b):
    return x * w + b

# --- Loss function (Mean Squared Error) ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --- Training loop ---
for epoch in range(20):
    # Forward pass
    y_pred = forward(x, w, b)
    loss = mse(y_true, y_pred)

    # Gradients (derivatives)
    dloss_dypred = 2 * (y_pred - y_true)
    dypred_dw = x
    dypred_db = 1

    dloss_dw = dloss_dypred * dypred_dw
    dloss_db = dloss_dypred * dypred_db

    # Update weights
    w -= lr * dloss_dw
    b -= lr * dloss_db

    print(f"Epoch {epoch+1:02d}: loss={loss:.4f}, w={w[0]:.4f}, b={b[0]:.4f}")

print(f"Final prediction: {forward(x, w, b)[0]:.4f}")
