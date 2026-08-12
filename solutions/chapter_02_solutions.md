# Chapter 2 Solutions: Mathematical Foundations

### Exercise 2.1. Transpose Reversal
**Question:** Prove $(AB)^T = B^T A^T$ using index notation.

**Solution:**
Let $C = AB$. By definition of matrix multiplication, the element in row $i$ and column $j$ of $C$ is:
$$C_{ij} = \sum_k A_{ik} B_{kj}$$
The $(i, j)$ entry of $C^T$ is simply $C_{ji}$:
$$(C^T)_{ij} = C_{ji} = \sum_k A_{jk} B_{ki}$$
Now, consider the matrix product $B^T A^T$. Its $(i, j)$ entry is the dot product of the $i$-th row of $B^T$ and the $j$-th column of $A^T$:
$$(B^T A^T)_{ij} = \sum_k (B^T)_{ik} (A^T)_{kj} = \sum_k B_{ki} A_{jk}$$
Since scalar multiplication is commutative, $\sum_k B_{ki} A_{jk} = \sum_k A_{jk} B_{ki}$. Therefore, $(AB)^T = B^T A^T$.

---

### Exercise 2.2. Projection Matrix
**Question:** Show that $H = X(X^T X)^{-1} X^T$ is symmetric, idempotent, and state its eigenvalues and rank.

**Solution:**
*   **(a) Symmetric ($H = H^T$):**
    $$H^T = (X(X^T X)^{-1} X^T)^T$$
    Applying the transpose reversal property:
    $$H^T = (X^T)^T ((X^T X)^{-1})^T X^T$$
    Since $X^T X$ is symmetric, its inverse is also symmetric. Thus:
    $$H^T = X (X^T X)^{-1} X^T = H$$
*   **(b) Idempotent ($H^2 = H$):**
    $$H^2 = (X(X^T X)^{-1} X^T) (X(X^T X)^{-1} X^T)$$
    Group the middle terms:
    $$H^2 = X(X^T X)^{-1} (X^T X) (X^T X)^{-1} X^T$$
    Since $(X^T X)^{-1} (X^T X) = I$:
    $$H^2 = X I (X^T X)^{-1} X^T = X(X^T X)^{-1} X^T = H$$
*   **(c) Eigenvalues and Rank:**
    The eigenvalues of any idempotent matrix are strictly $0.0$ or $1.0$. The rank of an idempotent matrix is equal to its trace. Assuming $X \in \mathbb{R}^{n \times (d+1)}$, the rank of $H$ is $d + 1.0$.

---

### Exercise 2.6. MLE Derivations
**Question:** Derive the MLE for an Exponential distribution given i.i.d. observations $x^{(1)}, ..., x^{(n)}$.

**Solution:**
The PDF of an Exponential distribution is $p(x; \lambda) = \lambda e^{-\lambda x}$.
The likelihood function $L(\lambda)$ is:
$$L(\lambda) = \prod_{i=1}^n \lambda e^{-\lambda x^{(i)}}$$
The log-likelihood $\ell(\lambda)$ converts the product to a sum:
$$\ell(\lambda) = \sum_{i=1}^n (\ln \lambda - \lambda x^{(i)}) = n \ln \lambda - \lambda \sum_{i=1}^n x^{(i)}$$
To find the maximum, compute the derivative with respect to $\lambda$ and set it to $0.0$:
$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x^{(i)} = 0.0$$
$$\frac{n}{\lambda} = \sum_{i=1}^n x^{(i)}$$
$$\hat{\lambda}_{MLE} = \frac{n}{\sum_{i=1}^n x^{(i)}} = \frac{1.0}{\bar{x}}$$
The MLE for the rate parameter $\lambda$ is exactly the inverse of the sample mean.
