
import numpy as np
from numpy.linalg import matrix_rank
import random
import scipy
from scipy import sparse
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

class Node:
    def __init__(self):
        self.rank = None
        self.size = None
        self.singular_values = None
        self.U = None
        self.V = None
        self.sons = None

def bound(a, b, value):
    if a > b:
        a, b = b, a
    return max(a, min(value, b))

def compare(A, B):
    return np.allclose(np.array(A, dtype=float), np.array(B, dtype=float), rtol=1e-05,
                           atol=1e-08, equal_nan=False)
def create_M(n):
    return np.array([[(random.uniform(0.00000001, 1)) for _ in range(n)] for _ in range(n)])
def create_sparse(n, density):
    rng = np.random.default_rng()
    A = sparse.random(n, n, density=density, random_state=rng)
    return np.array(A.A)
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
def find_s_index_for_delta(s, delta):
    if s.size == 0:
        return 0
    if s[0] <= delta:
        return 0
    for i in range(len(s)):
        if s[i] <= delta:
            return i
    return len(s)-1
def truncated_SVD(A, delta, b):
    b = b-1
    U, s, V = np.linalg.svd(A)
    i = find_s_index_for_delta(s, delta)
    # i = b
    idx = min(i, b, len(s))
    return U[:, :idx + 1], s[:idx+1], V[:idx + 1, :]
    # return randomized_svd(A, n_components=b)
def show_array(repr, zeros=False, show=False):
    if zeros:
        plot = plt.imshow((repr != 0).astype(int), cmap='Greys')
    else:
        plot = plt.imshow(repr, cmap='Greys')
    if show:
        plt.colorbar()
        plt.show()
    return plot

def show_arrays(repr1, repr2, zeros=False):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if zeros:
        plt.imshow((repr1 != 0).astype(int), cmap='Greys')
    else:
        plt.imshow(repr1, cmap='Greys')
    plt.colorbar()
    plt.title('Array 1')

    # Plot the second array
    plt.subplot(1, 2, 2)
    if zeros:
        plt.imshow((repr2 != 0).astype(int), cmap='Greys')
    else:
        plt.imshow(repr2, cmap='Greys')
    plt.colorbar()
    plt.title('Array 2')

    plt.tight_layout()
    plt.show()

def U_V_to_array(U, V):
    n = U.shape[0]
    m = U.shape[1]
    repr = np.zeros((n, n))
    repr[:n, :m] = U
    repr[:m, :n] = V
    return repr
def MSE(A, B):
    return ((A - B)**2).mean()



def compress_matrix(A, delta, b):
    if not np.any(A):
        v = Node()
        v.rank = 0
        v.size = A.shape
        return v
    U, s, V = truncated_SVD(A, delta, b)
    rank = matrix_rank(s)
    v = Node()
    v.rank = rank
    v.U = U * s
    v.V = V
    v.size = A.shape
    return v
def create_tree(A, r, e):
    r = bound(min(A.shape[0], A.shape[1]), 0, r+1)
    U, s, V = truncated_SVD(A, e, r + 1)
    if s[-1] < e or U.shape[0] <= r:
        v = compress_matrix(A, e, r)
    else:
        v = Node()
        A11, A12, A21, A22 = matrix_partition(A)
        v.sons = [create_tree(A11, r, e),
                  create_tree(A12, r, e),
                  create_tree(A21, r, e),
                  create_tree(A22, r, e)]
    return v
def recover_matrix(v):
    if v.sons:
        m = matrix_repartition(recover_matrix(v.sons[0]), recover_matrix(v.sons[1]),
                               recover_matrix(v.sons[2]), recover_matrix(v.sons[3]))
    elif v.rank == 0:
        m = np.zeros(v.size)
    else:
        m = v.U @ v.V
    return m
def tree_to_repr(v):
    if v.sons:
        m = matrix_repartition(tree_to_repr(v.sons[0]), tree_to_repr(v.sons[1]),
                               tree_to_repr(v.sons[2]), tree_to_repr(v.sons[3]))
    elif v.rank == 0:
        m = np.zeros(v.size)
    else:
        m = U_V_to_array(v.U, v.V)
    return m

def tree_to_repr_iterative(root):
    stack = [(root, None)]  # Initialize stack with root and its partially completed result (None initially)
    result = {}
    processing_order = []  # Store the order of processing
    partial_plots = []  # Store all partial plots

    while stack:
        node, partial_result = stack.pop()

        if node.sons:
            if node not in result:
                # Save the current node and its partially completed result on the stack
                # stack.append((node, partial_result))
                for son in reversed(node.sons):
                    stack.append((son, None))  # Initialize sons with None as partial result
            else:
                A11 = result[node.sons[0]]
                A12 = result[node.sons[1]]
                A21 = result[node.sons[2]]
                A22 = result[node.sons[3]]
                result[node] = matrix_repartition(A11, A12, A21, A22)
                processing_order.append((node, result[node]))  # Store the node and its result
                # partial_plots.append(show_array(result[node]))  # Store the plot
        elif node.rank == 0:
            result[node] = np.zeros(node.size)
            processing_order.append((node, result[node]))  # Store the node and its result
            # partial_plots.append(show_array(result[node]))  # Store the plot
        else:
            result[node] = U_V_to_array(node.U, node.V)
            processing_order.append((node, result[node]))  # Store the node and its result
            # partial_plots.append(show_array(result[node]))  # Store the plot

        # Update partially completed result for the parent node
        if partial_result is not None:
            result[node] = partial_result

    # Display all partial results
    for node, partial_result in processing_order:
        partial_plots.append(show_array(partial_result))
        # print(f"Node {node} - Partial Result:\n{partial_result}\n")

    return result[root], partial_plots



A = create_sparse(2**3, 0.02)
show_array(A, True, True)


U, s, V = np.linalg.svd(A)
plt.plot(s)
plt.show()

sigma_two = s[1]
sigma_half = s[len(s)//2]
sigma_5_7 = s[5*len(s)//7]
sigma_last = s[-1]

tree = create_tree(A, 0, sigma_last)
B = recover_matrix(tree)
show_array(tree_to_repr(tree), True, True)
compare(A, B)

show_arrays(A, B, True)

MSE(A, B)

import imageio

# Your function that generates the partial plots and returns partial_plots
result, partial_plots = tree_to_repr_iterative(tree)  # Assuming you have already defined 'root'

# Save partial plots as frames of a GIF
# imageio.mimsave('partial_plots.gif', partial_plots, duration=0.5)