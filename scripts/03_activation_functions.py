import numpy as np

# --- Data ---
X = np.array([[0.6, 0.2, 0.9]])
y_true = np.array([[1.0, 0.0]])

# --- Initialize weights and biases ---
W = np.random.randn(3, 2)
b = np.random.randn(1, 2)
lr = 0.1

# --- Activation functions and their derivatives ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- Forward function (with activation) ---
def forward(X, W, b, activation="relu"):
    z = np.dot(X, W) + b
    if activation == "relu":
        return relu(z), z
    elif activation == "sigmoid":
        return sigmoid(z), z

# --- Training loop ---
losses = []
for epoch in range(30):
    # Forward pass
    y_pred, z = forward(X, W, b, activation="relu")
    loss = np.mean((y_true - y_pred) ** 2)
    losses.append(loss)

    # Backward pass
    dL_dy = 2 * (y_pred - y_true)
    dy_dz = relu_derivative(z)
    dL_dz = dL_dy * dy_dz

    dL_dW = np.dot(X.T, dL_dz)
    dL_db = dL_dz

    W -= lr * dL_dW
    b -= lr * dL_db

    print(f"Epoch {epoch+1:02d}: loss={loss:.4f}")

print("\nFinal predictions:", y_pred)
