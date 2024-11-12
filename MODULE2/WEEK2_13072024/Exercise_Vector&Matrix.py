import numpy as np
import math


def compute_vector_length(vector):
    vector_length = math.sqrt(np.sum(vector**2))
    return vector_length


def compute_dot_product(vector1, vector2):
    dot_product = np.einsum('i,i->', vector1, vector2)
    return dot_product


def matrix_multi_vector(matrix, vector):
    result = np.einsum('ij,j->i', matrix, vector)
    return result


def matrix_multi_matrix(matrix1, matrix2):
    result = np.einsum('ij,jk->ik', matrix1, matrix2)
    return result


def inverse_matrix(matrix):
    det = matrix[0, 0]*matrix[1, 1] - matrix[0, 1]*matrix[1, 0]
    if det == 0:
        print("e) Matrix A is not invertible.")
        return
    inv_mtrx = np.linalg.inv(matrix)
    print(f"e) Matrix Inverse of Matrix A: {inv_mtrx}")


if __name__ == "__main__":
    # a: do dai cua vector
    my_vector = np.array([-2, 4, 9, 21])
    print(f"a) Vector Length: {compute_vector_length(my_vector)}")

    # b: phep tich vo huong
    vector1 = np.array([0, 1, -1, 2])
    vector2 = np.array([2, 5, 1, 0])
    print(
        f"b) Dot product of vector1 and vector2 is: {compute_dot_product(vector1,vector2)}")

    # c: nhan vector voi ma tran
    matrix_c = np.array([[-1, 1, 1],
                         [0, -4, 9]])
    vector_c = np.array([0, 2, 1])
    print(
        f"c) The multiplication of matrix and vector: {matrix_multi_vector(matrix_c,vector_c)}")

    # d: nhan ma tran voi ma tran
    matrix_d1 = np.array([[0, 1, 2],
                          [2, -3, 1]])
    matrix_d2 = np.array([[1, -3],
                          [6, 1],
                          [0, -1]])
    print(
        f"d) The multiplication of matrix and matrix: {matrix_multi_matrix(matrix_d1,matrix_d2)}")

    # e: ma tran nghich dao
    matrix_e = np.array([[1, 2],
                         [3, 4]])
    inverse_matrix(matrix_e)
