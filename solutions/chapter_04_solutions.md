# Chapter 4 Solutions: Regression

### Exercise 4.1. Normal Equations from scratch
**Question:** Starting from $J(\theta) = \frac{1}{2n} \|y - X\theta\|_2^2$[cite: 1], derive the Normal Equations using the matrix calculus identities $\nabla_\theta(a^T\theta) = a$ and $\nabla_\theta(\theta^T A \theta) = 2A\theta$ (when $A = A^T$)[cite: 1]. State clearly when the solution is unique.

**Solution:**
Step 1: Expand the squared norm (where $\|v\|_2^2 = v^Tv$)[cite: 1].
$$J(\theta) = \frac{1}{2n} (y - X\theta)^T (y - X\theta)$$
$$J(\theta) = \frac{1}{2n} (y^Ty - y^TX\theta - \theta^TX^Ty + \theta^TX^TX\theta)$$

Step 2: Recognize that $y^TX\theta$ is a scalar, so it equals its transpose $\theta^TX^Ty$[cite: 1]. Combine the middle terms:
$$J(\theta) = \frac{1}{2n} (y^Ty - 2\theta^TX^Ty + \theta^T(X^TX)\theta)$$

Step 3: Compute the gradient with respect to $\theta$. Note that $X^TX$ is a symmetric matrix.
$$\nabla_\theta J = \frac{1}{2n} \left( 0.0 - 2X^Ty + 2X^TX\theta \right)$$
$$\nabla_\theta J = \frac{1}{n} X^T(X\theta - y)$$

Step 4: Set the gradient to $0.0$ to find the minimum[cite: 1]:
$$\frac{1}{n} X^T(X\theta - y) = 0.0$$
$$X^TX\hat{\theta} = X^Ty$$

**Uniqueness Condition:**
The solution is unique if and only if the matrix $(X^TX)$ is invertible[cite: 1]. This requires the design matrix $X$ to have full column rank, meaning there is no perfect multicollinearity and the number of independent observations $n$ is greater than or equal to the number of features $d + 1.0$[cite: 1].
