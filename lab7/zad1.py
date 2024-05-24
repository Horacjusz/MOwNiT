import numpy as np
import mpmath as mp

def generate_matrix(n, a, b, precision):
    # Calculate the number of elements in the matrix
    num_elements = n * n
    
    # Generate random numbers between a and b with specified precision
    random_values = np.random.uniform(a, b, num_elements)
    
    # Round the values to the specified precision
    rounded_values = np.round(random_values, precision)
    
    # Reshape the 1D array into an n x n matrix
    matrix = rounded_values.reshape(n, n)
    
    return matrix

def power_method(matrix, eps = 1e-10, iterations = 10000) :
    n = matrix.shape[0]
    vect = np.zeros(n)
    vect[0] = 1
    # vect[1] = 1
    
    value = 0
    for _ in range(iterations) :
        vect = np.dot(matrix,vect)
        value = np.max(np.abs(vect))
        vect = vect / np.linalg.norm(vect)
    
    return vect,value
        

# Wartości parametrów
n = 5  # Rozmiar macierzy nxn
a = 0  # Dolna granica zakresu losowych liczb
b = 1  # Górna granica zakresu losowych liczb
prec = 33

# Wygenerowanie macierzy
matrix = np.array([[0.74332056, 0.27598035, 0.60583225, 0.57048321, 0.56251498],
 [0.13529092, 0.67841427, 0.56881017, 0.08723244, 0.84103551],
 [0.37385615, 0.60823516, 0.94620079, 0.67527718, 0.20754133],
 [0.37019073, 0.54651661, 0.96760486, 0.02035754, 0.98166398],
 [0.42413919, 0.84570564, 0.398043,   0.08447961, 0.97703725]])


print(matrix)

vect1,value = power_method(matrix,prec)
print(vect1)
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(eigenvectors[0])
