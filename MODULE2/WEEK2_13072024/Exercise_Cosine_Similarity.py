import math
import numpy as np


def compute_cosine(v1, v2):
    dot_product = np.einsum('i,i->', v1, v2)
    v1_length = np.sum(v1**2)
    v2_length = np.sum(v2**2)
    cos_sim = dot_product / (math.sqrt(v1_length)*math.sqrt(v2_length))
    return cos_sim


if __name__ == "__main__":
    vector_x = np.array([1, 2, 3, 4])
    vector_y = np.array([1, 0, 3, 0])
    print(compute_cosine(vector_x, vector_y))
