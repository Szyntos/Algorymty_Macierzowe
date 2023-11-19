import numpy as np
from collections import Counter
import pandas as pd
import random
from pprint import pprint


class Num(float):
    counts = Counter(mul=0, add=0)

    def __add__(self, other):
        self.counts["add"] += 1
        return Num(super(Num, self).__add__(other))

    def __mul__(self, other):
        self.counts["mul"] += 1
        return Num(super(Num, self).__mul__(other))

    def __sub__(self, other):
        self.counts["add"] += 1
        return Num(super(Num, self).__sub__(other))

    def reset_counter(self):
        self.counts["mul"] = 0
        self.counts["add"] = 0


def matrix_partition(A):
    n = A.shape[0] // 2
    m = A.shape[1] // 2
    A11 = A[:n, :m]
    A12 = A[:n, m:]
    A21 = A[n:, :m]
    A22 = A[n:, m:]
    return A11, A12, A21, A22


def matrix_repartition(A11, A12, A21, A22):
    C = np.block([[A11, A12],
                  [A21, A22]])
    return C


def compare(A, B):
    return np.allclose(np.array(A, dtype=float), np.array(B, dtype=float), rtol=1e-05,
                       atol=1e-08, equal_nan=False)


def create_M_2(n):
    n = 2 ** n
    return np.array([[Num((random.uniform(0.00000001, 1))) for _ in range(n)] for _ in range(n)], dtype=Num)


def create_M(n):
    return np.array([[Num((random.uniform(0.00000001, 1))) for _ in range(n)] for _ in range(n)], dtype=Num)


def inverse(A):
    if A.shape[0] == 1:
        if A[0][0] != 0:
            return np.array([[1 / A[0][0]]])
        return np.array([[A[0][0]]])
    A11, A12, A21, A22 = matrix_partition(A)
    A11_i = inverse(A11)
    S22 = A22 - A21 @ A11_i @ A12
    S22_i = inverse(S22)
    B11 = A11_i @ (np.eye(A11_i.shape[0], dtype=Num) + A12 @ S22_i @ A21 @ A11_i)
    B12 = -A11_i @ A12 @ S22_i
    B21 = -S22_i @ A21 @ A11_i
    B22 = S22_i
    return matrix_repartition(B11, B12, B21, B22)


def check_inverse(A):
    return compare(inverse(A), np.linalg.inv(np.array(A, dtype=float)))


A = create_M(7)
print(check_inverse(A))


def LU(A):
    if A.shape[0] == 1:
        return np.array(A), np.array([[1]])
    A11, A12, A21, A22 = matrix_partition(A)
    L11, U11 = LU(A11)
    U11_i = inverse(U11)
    L21 = A21 @ U11_i
    L11_i = inverse(L11)
    U12 = L11_i @ A12
    S = A22 - A21 @ U11_i @ L11_i @ A12
    Ls, Us = LU(S)
    U22 = Us
    L22 = Ls
    return [matrix_repartition(L11, np.zeros((L11.shape[0], L22.shape[1]), dtype=Num), L21, L22),
            matrix_repartition(U11, U12, np.zeros((U22.shape[0], U11.shape[1]), dtype=Num), U22)]


def check_LU(A):
    L, U = LU(A)
    return compare(L, np.tril(L)) and compare(U, np.triu(U)) and compare(A, L @ U)


A = create_M(3)
L, U = LU(A)
print(check_LU(A))


def determinant(A):
    L, U = LU(A)
    return np.prod([L[i][i] for i in range(L.shape[0])]) * np.prod([U[i][i] for i in range(U.shape[0])])


def check_determinant(A):
    return compare(determinant(A), np.linalg.det(np.array(A, dtype=float)))


print(check_determinant(A))
