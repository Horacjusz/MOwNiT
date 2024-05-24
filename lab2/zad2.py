import numpy as np
from copy import deepcopy
import time
import matplotlib.pyplot as plt


def LU_factorization(A):
    n = A.shape[0]
    L = np.eye(n)
    for k in range(n-1):
        if A[k, k] == 0:
            max_row_index = np.argmax(np.abs(A[k+1:, k])) + k + 1
            A[[k, max_row_index]] = A[[max_row_index, k]]
            L[[k, max_row_index], :k] = L[[max_row_index, k], :k]
            
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            L[i, k] = factor
            
            for j in range(k,n) :
                A[i,j] -= factor * A[k,j]
            
            # A[i, k+1:] -= factor * A[k, k+1:]
    return L, A

def matrix_multiplication(A, B):
    if A.shape[1] != B.shape[0]:
        return None
    
    return np.dot(A, B)
        
        

def frobenius_norm_difference(A, L, U):
    return np.linalg.norm(A - np.dot(L, U))

# Wygenerowanie tablicy n liczb całkowitych z zakresu 2 do 500
n_values = np.random.randint(2, 500, size=100)
n_values.sort()

def generate_random_matrix(n):
    return np.random.randint(0, 500, size=(n, n)).astype(np.float64)

eps = 10**-5

times = []

for i in range(len(n_values)):
    n = n_values[i]
    print("\nTest nr:",i,"\nSIZE:",n)
    start_time = time.time()
    A = generate_random_matrix(n)
    org = deepcopy(A)
    L, U = LU_factorization(A)
    fr = frobenius_norm_difference(org, L, U)
    end_time = time.time()
    
    times.append(end_time - start_time)
    
    if fr > eps : 
        print("ERROR",fr)
    print("Finished")
    
    
plt.figure(figsize=(10, 6))
plt.plot(n_values, times, label='Exercise implementation', color='blue')
# plt.plot(sizes, times_numpy, label='NumPy', color='red')
plt.title('Czas działania algorytmu w zależności od rozmiaru macierzy')
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas [s]')
plt.legend()
plt.savefig('wykres2.png')
plt.show()