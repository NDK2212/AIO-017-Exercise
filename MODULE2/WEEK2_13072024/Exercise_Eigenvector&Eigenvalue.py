import math
import numpy as np

a = np.array([[0.9, 0.2],
              [0.1, 0.8]])
eigenvalues, eigenvectors = np.linalg.eig(a)
print(f"Eigenvectors: {eigenvectors}")
normalised_eigenvectors = eigenvectors / \
    np.linalg.norm(eigenvectors, axis=1)
print(f"Eigenvalues: {eigenvalues}")
print(f"Normalised Eigenvectors: {normalised_eigenvectors}")
