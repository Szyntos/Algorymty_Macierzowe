import numpy as np
from collections import Counter
import pandas as pd
import random
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


class MatrixMultiplier:

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.result = np.zeros((1, 1), dtype=A.dtype)
        filename = 'factorizations_r.npz'
        with open(filename, 'rb') as f:
            self.factorizations = dict(np.load(f, allow_pickle=True))

    def is_power_of_two(self, n):
        if n == 1 or n == 0:
            return 0
        return (n & (n - 1)) == 0

    def matrix_partition_sq2(self, A):
        n = A.shape[0] // 2
        m = A.shape[1] // 2
        A11 = A[:n, :m]
        A12 = A[:n, m:]
        A21 = A[n:, :m]
        A22 = A[n:, m:]
        return A11, A12, A21, A22

    def matrix_repartition_sq2(self, A11, A12, A21, A22):
        C = np.block([[A11, A12],
                      [A21, A22]])
        return C

    def iterative_wrap(self, A, B):
        C = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                s = Num(0)
                for k in range(B.shape[0]):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        self.result = C
        return C

    def binet_wrap(self, A, B):
        C = np.zeros((1, 1), dtype=A.dtype)

        if A.shape[0] == A.shape[1] and A.shape[1] == B.shape[0] and B.shape[0] == B.shape[1]:
            if A.shape[0] == 1:
                return np.array([[Num(A[0, 0] * B[0, 0])]], dtype=Num)
            elif self.is_power_of_two(B.shape[0]):
                A11, A12, A21, A22 = self.matrix_partition_sq2(A)
                B11, B12, B21, B22 = self.matrix_partition_sq2(B)
                C = self.matrix_repartition_sq2(self.binet_wrap(A11, B11) + self.binet_wrap(A12, B21),
                                                self.binet_wrap(A11, B12) + self.binet_wrap(A12, B22),
                                                self.binet_wrap(A21, B11) + self.binet_wrap(A22, B21),
                                                self.binet_wrap(A21, B12) + self.binet_wrap(A22, B22))

        return C

    def strassen_wrap(self, A, B):
        C = np.zeros((1, 1), dtype=A.dtype)

        if A.shape[0] == A.shape[1] and A.shape[1] == B.shape[0] and B.shape[0] == B.shape[1]:
            if A.shape[0] == 1:
                return np.array([[Num(A[0, 0] * B[0, 0])]], dtype=Num)
            elif B.shape[0] > 1 and (B.shape[0] & (B.shape[0] - 1)) == 0:
                A11, A12, A21, A22 = self.matrix_partition_sq2(A)
                B11, B12, B21, B22 = self.matrix_partition_sq2(B)
                P1 = self.strassen_wrap(A11 + A22, B11 + B22)
                P2 = self.strassen_wrap(A21 + A22, B11)
                P3 = self.strassen_wrap(A11, B12 - B22)
                P4 = self.strassen_wrap(A22, B21 - B11)
                P5 = self.strassen_wrap(A11 + A12, B22)
                P6 = self.strassen_wrap(A21 - A11, B11 + B12)
                P7 = self.strassen_wrap(A12 - A22, B21 + B22)

                C = self.matrix_repartition_sq2(P1 + P4 - P5 + P7, P3 + P5,
                                                P2 + P4, P1 - P2 + P3 + P6)

        return C

    def algorithm_from_factors(self, factors: np.ndarray, n, m, k):
        factors = [factors[0].copy(), factors[1].copy(), factors[2].copy()]
        rank = factors[0].shape[-1]
        factors[0] = factors[0].reshape(n, m, rank)
        factors[1] = factors[1].reshape(m, k, rank)
        factors[2] = factors[2].reshape(k, n, rank)
        factors[2] = factors[2].transpose(1, 0, 2)

        def f(a, b):
            n = a.shape[0]
            m = a.shape[1]
            l = b.shape[1]

            result = np.array([[0 for _ in range(l)] for _ in range(n)], dtype=Num)
            for alpha in range(rank):
                left = Num()
                for i in range(n):
                    for j in range(m):
                        if factors[0][i, j, alpha] != 0:
                            curr = factors[0][i, j, alpha] * a[i][j]
                            left += curr
                right = Num()
                for j in range(m):
                    for k in range(l):
                        if factors[1][j, k, alpha] != 0:
                            curr = factors[1][j, k, alpha] * b[j][k]
                            right += curr
                matrix_product = left * right
                for i in range(n):
                    for k in range(l):
                        if factors[2][i, k, alpha] != 0:
                            curr = factors[2][i, k, alpha] * matrix_product
                            result[i, k] += curr
            return result

        return f

    def iterative(self):
        self.result = self.iterative_wrap(self.A, self.B)
        return self.result

    def strassen(self):
        self.result = self.strassen_wrap(self.A, self.B)
        return self.result

    def binet(self):
        self.result = self.binet_wrap(self.A, self.B)
        return self.result

    def alphatensor(self, A, B):
        assert A.shape[1] == B.shape[0]
        n = A.shape[0]
        m = A.shape[1]
        k = B.shape[1]
        factors = self.factorizations[str(n) + ',' + str(m) + ',' + str(k)]
        matrix_mul_algorithm = self.algorithm_from_factors(factors, n, m, k)
        return matrix_mul_algorithm(A, B)

    def alpha_strassen_wrap(self, A, B):
        C = np.zeros((1, 1), dtype=A.dtype)

        if (str(A.shape[0]) + ',' + str(A.shape[1]) + ',' + str(B.shape[1])) in self.factorizations:
            return self.alphatensor(A, B)
        elif A.shape[0] == 1:
            return np.array([[Num(A[0, 0] * B[0, 0])]], dtype=Num)
        else:
            A11, A12, A21, A22 = self.matrix_partition_sq2(A)
            B11, B12, B21, B22 = self.matrix_partition_sq2(B)
            P1 = self.alpha_strassen_wrap(A11 + A22, B11 + B22)
            P2 = self.alpha_strassen_wrap(A21 + A22, B11)
            P3 = self.alpha_strassen_wrap(A11, B12 - B22)
            P4 = self.alpha_strassen_wrap(A22, B21 - B11)
            P5 = self.alpha_strassen_wrap(A11 + A12, B22)
            P6 = self.alpha_strassen_wrap(A21 - A11, B11 + B12)
            P7 = self.alpha_strassen_wrap(A12 - A22, B21 + B22)

            C = self.matrix_repartition_sq2(P1 + P4 - P5 + P7, P3 + P5,
                                            P2 + P4, P1 - P2 + P3 + P6)

        return C
    def alpha_binet_wrap(self, A, B):
        C = np.zeros((1, 1), dtype=A.dtype)
        if (str(A.shape[0]) + ',' + str(A.shape[1]) + ',' + str(B.shape[1])) in self.factorizations:
            return self.alphatensor(A, B)
        elif A.shape[0] == 1:
            return np.array([[Num(A[0, 0] * B[0, 0])]], dtype=Num)
        else:
            A11, A12, A21, A22 = self.matrix_partition_sq2(A)
            B11, B12, B21, B22 = self.matrix_partition_sq2(B)
            C = self.matrix_repartition_sq2(self.alpha_binet_wrap(A11, B11) + self.alpha_binet_wrap(A12, B21),
                                            self.alpha_binet_wrap(A11, B12) + self.alpha_binet_wrap(A12, B22),
                                            self.alpha_binet_wrap(A21, B11) + self.alpha_binet_wrap(A22, B21),
                                            self.alpha_binet_wrap(A21, B12) + self.alpha_binet_wrap(A22, B22))

        return C
    def alpha_strassen(self):
        self.result = np.array(self.alpha_strassen_wrap(self.A, self.B), dtype=Num)
        return self.result

    def alpha_binet(self):
        self.result = np.array(self.alpha_binet_wrap(self.A, self.B), dtype=Num)
        return self.result

    def reset_counter(self):
        Num().reset_counter()

    def check(self):
        return np.allclose(np.array(self.result, dtype=float), np.array(self.A @ self.B, dtype=float), rtol=1e-05,
                           atol=1e-08, equal_nan=False)

    def get_current_flops(self):
        return Num().counts


def create_M_2(n):
    n = 2 ** n
    return np.array([[Num((j * n + i + 1)/(n*n)) for i in range(n)] for j in range(n)], dtype=Num)
A = create_M_2(4)
B = create_M_2(4)
M = MatrixMultiplier(A, B)
M.reset_counter()
M.strassen()
print(M.get_current_flops())
M.reset_counter()
M.binet()
print(M.get_current_flops())
x = 2**3
C = np.array([[Num((random.uniform(0.00000001, 1))) for i in range(4*x)] for j in range(4*x)], dtype=Num)
D = np.array([[Num((random.uniform(0.00000001, 1))) for i in range(5*x)] for j in range(4*x)], dtype=Num)
M.A = C
M.B = D
print(M.B.shape)
M.reset_counter()
M.alpha_binet()
print(M.get_current_flops())
M.reset_counter()
M.iterative()
print(M.get_current_flops())
print(M.check())

def test_functions(bound):
    df = pd.DataFrame()
    MM = MatrixMultiplier(create_M_2(1), create_M_2(1))
    for n in range(2, bound):
        MM.A, MM.B = create_M_2(n), create_M_2(n)
        methods = {"Iterative":MM.iterative, "Binet":MM.binet, "Strassen":MM.strassen, "AlphaStrassen": MM.alpha_strassen}
        for method_name, method in methods.items():
            print(n, method_name)
            method()
            df.at[n, method_name] = MM.check()
    return df

test_functions(3)
