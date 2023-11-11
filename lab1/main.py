from Matrix import Matrix


class MatrixMultiplier:

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.iterative_flops = {'add': 0, 'sub': 0, 'mul': 0, 'div': 0}
        self.Binet_flops = {'add': 0, 'sub': 0, 'mul': 0, 'div': 0}
        self.Strassen_flops = {'add': 0, 'sub': 0, 'mul': 0, 'div': 0}

    def is_power_of_two(self, n):
        if n == 1 or n == 0:
            return 0
        return (n & (n - 1)) == 0

    def matrix_partition_sq2(self, A):
        n = int(A.n / 2)
        A11 = Matrix([[A[i][j] for j in range(n)] for i in range(n)])
        A12 = Matrix([[A[i][j + n] for j in range(n)] for i in range(n)])
        A21 = Matrix([[A[i + n][j] for j in range(n)] for i in range(n)])
        A22 = Matrix([[A[i + n][j + n] for j in range(n)] for i in range(n)])
        return A11, A12, A21, A22

    def matrix_repartition_sq2(self, A11, A12, A21, A22):
        n = int(A11.n)
        A = Matrix([[0 for _ in range(n * 2)] for _ in range(n * 2)])
        for i in range(n):
            for j in range(n):
                A[i][j] = A11[j][i]
                A[i + n][j] = A21[j][i]
                A[i][j + n] = A12[j][i]
                A[i + n][j + n] = A22[j][i]
        return A

    def iterative_wrap(self, A, B):
        C = Matrix([[0]])
        if A.n == A.m and A.n == B.n and B.n == B.m:
            C = Matrix([[0 for _ in range(A.n)] for _ in range(B.m)])
            for i in range(A.n):
                for j in range(B.m):
                    s = 0
                    for k in range(B.m):
                        self.iterative_flops['mul'] += 1
                        self.iterative_flops['add'] += 1
                        s += A[i][k] * B[k][j]
                    C[j][i] = s
        return C

    def Binet_wrap(self, A, B):
        C = Matrix([[0]])
        if A.n == A.m and A.n == B.n and B.n == B.m:
            if A.n == 1:
                self.Binet_flops['mul'] += 1
                return Matrix([[A[0][0] * B[0][0]]])
            elif self.is_power_of_two(B.n):
                A11, A12, A21, A22 = self.matrix_partition_sq2(A)
                B11, B12, B21, B22 = self.matrix_partition_sq2(B)
                self.Binet_flops['add'] += 4 * A11.n * A11.n
                C = self.matrix_repartition_sq2(self.Binet_wrap(A11, B11) + self.Binet_wrap(A12, B21),
                                                self.Binet_wrap(A11, B12) + self.Binet_wrap(A12, B22),
                                                self.Binet_wrap(A21, B11) + self.Binet_wrap(A22, B21),
                                                self.Binet_wrap(A21, B12) + self.Binet_wrap(A22, B22))
        return C

    def Strassen_wrap(self, A, B):
        C = Matrix([[0]])
        if A.n == A.m and A.n == B.n and B.n == B.m:
            if A.n == 1:
                self.Strassen_flops['mul'] += 1
                return Matrix([[A[0][0] * B[0][0]]])
            elif self.is_power_of_two(B.n):
                A11, A12, A21, A22 = self.matrix_partition_sq2(A)
                B11, B12, B21, B22 = self.matrix_partition_sq2(B)
                self.Strassen_flops['add'] += A11.n * A11.n
                self.Strassen_flops['add'] += B11.n * B11.n
                P1 = self.Strassen_wrap(A11 + A22, B11 + B22)
                self.Strassen_flops['add'] += A11.n * A11.n
                P2 = self.Strassen_wrap(A21 + A22, B11)
                self.Strassen_flops['sub'] += B12.n * B12.n
                P3 = self.Strassen_wrap(A11, B12 - B22)
                self.Strassen_flops['sub'] += B21.n * B21.n
                P4 = self.Strassen_wrap(A22, B21 - B11)
                self.Strassen_flops['add'] += A11.n * A11.n
                P5 = self.Strassen_wrap(A11 + A12, B22)
                self.Strassen_flops['sub'] += A21.n * A21.n
                self.Strassen_flops['add'] += B11.n * B11.n
                P6 = self.Strassen_wrap(A21 - A11, B11 + B12)
                self.Strassen_flops['sub'] += A12.n * A12.n
                self.Strassen_flops['add'] += B21.n * B21.n
                P7 = self.Strassen_wrap(A12 - A22, B21 + B22)
                self.Strassen_flops['add'] += P1.n * P1.n
                self.Strassen_flops['add'] += P5.n * P5.n
                self.Strassen_flops['add'] += P3.n * P3.n
                self.Strassen_flops['add'] += P2.n * P2.n
                self.Strassen_flops['add'] += P2.n * P2.n
                self.Strassen_flops['add'] += P3.n * P3.n
                self.Strassen_flops['sub'] += P4.n * P4.n
                self.Strassen_flops['sub'] += P1.n * P1.n
                C = self.matrix_repartition_sq2(P1 + P4 - P5 + P7,
                                                P3 + P5,
                                                P2 + P4,
                                                P1 - P2 + P3 + P6)
        return C

    def iterative(self):
        return self.iterative_wrap(self.A, self.B)

    def Strassen(self):
        return self.Strassen_wrap(self.A, self.B)

    def Binet(self):
        return self.Binet_wrap(self.A, self.B)

    def print_flops(self):
        print("iterative:\tsum:", sum(self.iterative_flops.values()), self.iterative_flops)
        print("Binet:  \tsum:", sum(self.Binet_flops.values()), self.Binet_flops)
        print("Strassen:\tsum:", sum(self.Strassen_flops.values()), self.Strassen_flops)


def create_M_2(n):
    n = 2 ** n
    return Matrix([[(j * n) + i + 1 for i in range(n)] for j in range(n)])


# A = Matrix([[12, 1, 0, 0],
#             [0, 12, 1, 0],
#             [0, 0, 12, 1],
#             [0, 5, 0, 12]])
# B = Matrix([[1, 2, 3, 4],
#             [5, 6, 7, 8],
#             [9, 10, 11, 12],
#             [13, 14, 15, 16]])

A = create_M_2(2)
B = create_M_2(2)

MM = MatrixMultiplier(A, B)
# # A = create_M_2(3)
binet = MM.Binet()

stras = MM.Strassen()

it = MM.iterative()

MM.print_flops()
