The Euclidean norm, also known as the $L^2$ norm or the $\ell_2$ norm, is a measure of the magnitude (length) of a vector in Euclidean space. It is the most commonly used norm in vector spaces and is defined as the square root of the sum of the squares of the vector's components.

### Definition

For a vector $\mathbf{x} = [x_1, x_2, \ldots, x_n] \in \mathbb{R}^n$, the Euclidean norm is given by:

$$\| \mathbf{x} \|_2 = \sqrt{ x_1^2 + x_2^2 + \cdots + x_n^2 }$$

### Properties

1. **Non-negativity**:
   $$
   \| \mathbf{x} \|_2 \geq 0
  $$
   The Euclidean norm is always non-negative, and it is zero if and only if the vector $\mathbf{x}$ is the zero vector.

2. **Homogeneity (Scaling)**:
   $$
   \| \alpha \mathbf{x} \|_2 = |\alpha| \| \mathbf{x} \|_2
  $$
   For any scalar $\alpha$, scaling the vector $\mathbf{x}$ by $\alpha$ scales the Euclidean norm by $|\alpha|$.

3. **Triangle Inequality**:
   $$
   \| \mathbf{x} + \mathbf{y} \|_2 \leq \| \mathbf{x} \|_2 + \| \mathbf{y} \|_2
  $$
   The Euclidean norm satisfies the triangle inequality, meaning the norm of the sum of two vectors is less than or equal to the sum of their norms.

### Example

Consider the vector $\mathbf{x}$:

$$\mathbf{x} = [3, 4]$$

The Euclidean norm of $\mathbf{x}$ is calculated as:

$$\| \mathbf{x} \|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

### Relation to the Frobenius Norm

For matrices, the Frobenius norm can be considered an extension of the Euclidean norm. Specifically, the Frobenius norm of a matrix is equivalent to the Euclidean norm of the vector formed by flattening the matrix into a single vector.

### Summary

The Euclidean norm is a way to measure the length or magnitude of a vector in Euclidean space. It is widely used in various mathematical and applied fields, including physics, engineering, computer science, and machine learning, to quantify the distance and similarity between vectors.