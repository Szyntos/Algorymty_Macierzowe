class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix[0])
        self.m = len(matrix)

    def __repr__(self):
        a = ""
        for row in self.matrix:
            for el in row:
                a += str(el) + "\t"
            a += "\n"
        return a

    def __add__(self, other):
        if self.n == other.n and self.m == other.m:
            return Matrix([[self[j][i] + other[j][i] for i in range(self.n)] for j in range(self.m)])
        else:
            raise Exception("Not matching sizes")

    def __sub__(self, other):
        if self.n == other.n and self.m == other.m:
            return Matrix([[self[j][i] - other[j][i] for i in range(self.n)] for j in range(self.m)])
        else:
            raise Exception("Not matching sizes")

    def __eq__(self, other):
        if not (self.n == other.n and self.m == other.m):
            return False
        for i in range(self.n):
            for j in range(self.m):
                if self[i][j] != other[i][j]:
                    return False
        return True

    def __getitem__(self, key):
        return self.matrix[key]
