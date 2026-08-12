import numpy as np
from typing import Tuple, List

class DenseLayer:
    """
    Fully connected layer with manual forward and backward passes.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        # He / Kaiming Normal Initialization for ReLU activations
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        
        # Caches for backpropagation
        self.x: np.ndarray = np.array([])
        self.dW: np.ndarray = np.array([])
        self.db: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Gradients w.r.t parameters
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        # Gradient w.r.t input
        dx = np.dot(dout, self.W.T)
        return dx

class ReLU:
    """
    Rectified Linear Unit activation.
    """
    def __init__(self) -> None:
        self.x: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout.copy()
        dx[self.x <= 0.0] = 0.0
        return dx

class SoftmaxCrossEntropyLoss:
    """
    Combined Softmax Activation and Cross-Entropy Loss for numerical stability.
    """
    def __init__(self) -> None:
        self.p: np.ndarray = np.array([])
        self.y_true: np.ndarray = np.array([])

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        self.y_true = y_true
        # Shift logits for numerical stability (prevent overflow)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        batch_size = logits.shape[0]
        # Cross-entropy loss computation
        log_likelihood = -np.log(self.p[np.arange(batch_size), y_true] + 1e-9)
        loss = np.sum(log_likelihood) / batch_size
        return float(loss)

    def backward(self) -> np.ndarray:
        batch_size = self.p.shape[0]
        dout = self.p.copy()
        # Derivative of Softmax + Cross Entropy simplifies to (p - y) / N
        dout[np.arange(batch_size), self.y_true] -= 1.0
        dout /= batch_size
        return dout

class MLPScratch:
    """
    Multi-Layer Perceptron built from modular layer primitives.
    """
    def __init__(self, layer_dims: List[int]) -> None:
        self.layers: List[DenseLayer] = []
        self.activations: List[ReLU] = []
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(DenseLayer(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                self.activations.append(ReLU())
                
        self.loss_fn = SoftmaxCrossEntropyLoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for i in range(len(self.layers) - 1):
            out = self.layers[i].forward(out)
            out = self.activations[i].forward(out)
        out = self.layers[-1].forward(out)
        return out

    def fit_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        # Forward Pass
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, y)
        
        # Backward Pass
        dout = self.loss_fn.backward()
        dout = self.layers[-1].backward(dout)
        
        for i in reversed(range(len(self.activations))):
            dout = self.activations[i].backward(dout)
            dout = self.layers[i].backward(dout)
            
        # Parameter Updates (SGD)
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db
            
        return loss

if __name__ == "__main__":
    np.random.seed(42)
    # Generate dummy multi-class dataset
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randint(0, 3, size=(100,))
    
    # 10 features -> 16 hidden -> 8 hidden -> 3 classes
    mlp = MLPScratch(layer_dims=[10, 16, 8, 3])
    
    print("--- Training MLP From Scratch ---")
    for epoch in range(1, 101):
        loss = mlp.fit_step(X_dummy, y_dummy, lr=0.1)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:d} Loss: {loss:.1f}")
