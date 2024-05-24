import numpy as np
from random import randint

def DFT(x, eps = 10**-15) :
    n = len(x)
    if n == 1 :
        return x
    
    r = int(np.log2(n))
    if 2 ** r != n :
        raise ValueError("Input vector length must be a power of 2")

    F = np.zeros((n, n), dtype=np.complex128)
    ksi = np.exp(-2.j * np.pi / n)
    for j in range(n) :
        for k in range(n) :
            val = ksi ** (j * k)
            if abs(val.real) < eps : val = complex(0, val.imag)
            if abs(val.imag) < eps : val = complex(val.real, 0)
            F[j, k] = val

    y = np.dot(F, x)
    return y

r = randint(1,3)

x = np.random.randint(0,10,size = 2**r)
y = DFT(x)
print("X:", x)
print("Y:", y)
print("Y:", np.fft.fft(x))