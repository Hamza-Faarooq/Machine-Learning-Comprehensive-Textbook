import numpy as np
import matplotlib.pyplot as plt

def plot_bias_variance_tradeoff() -> None:
    """
    Generates the Bias-Variance Tradeoff curve seen in the textbook[cite: 1].
    """
    complexities = np.linspace(1.0, 10.0, 100)
    
    # Mathematical relationships
    bias_squared = 10.0 / complexities
    variance = 0.5 * (complexities ** 1.5)
    irreducible_error = np.full_like(complexities, 2.0)
    total_error = bias_squared + variance + irreducible_error
    
    plt.figure(figsize=(8, 5))
    plt.plot(complexities, bias_squared, label="Bias^2", linestyle="--", color="blue")
    plt.plot(complexities, variance, label="Variance", linestyle="-.", color="red")
    plt.plot(complexities, total_error, label="Total Test Error", linewidth=2.0, color="purple")
    plt.axhline(y=2.0, color="gray", linestyle=":", label="Noise Floor (Var = 2.0)")
    
    plt.title("The Bias-Variance Tradeoff")
    plt.xlabel("Model Complexity")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("bias_variance_tradeoff.png", dpi=300)
    print("Saved bias_variance_tradeoff.png")

def plot_gradient_descent_1d() -> None:
    """
    Visualizes gradient descent finding the minimum of a convex bowl.
    """
    x = np.linspace(-5.0, 5.0, 100)
    y = x ** 2.0 # Simple convex quadratic cost function
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color="black", label="Cost Function J(theta)")
    
    # Simulate gradient descent steps
    theta = 4.0
    lr = 0.15
    for _ in range(5):
        cost = theta ** 2.0
        gradient = 2.0 * theta
        next_theta = theta - lr * gradient
        
        plt.scatter(theta, cost, color="red", zorder=5)
        plt.annotate("", xy=(next_theta, next_theta**2.0), xytext=(theta, cost),
                     arrowprops=dict(arrowstyle="->", color="blue"))
        theta = next_theta
        
    plt.title("Gradient Descent Optimization")
    plt.xlabel("Parameter (theta)")
    plt.ylabel("Cost J")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("gradient_descent_1d.png", dpi=300)
    print("Saved gradient_descent_1d.png")

if __name__ == "__main__":
    plot_bias_variance_tradeoff()
    plot_gradient_descent_1d()
