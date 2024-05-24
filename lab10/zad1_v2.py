import numpy as np

def DFT(x, eps=10**-15):
    n = len(x)
    if n == 1:
        return x
    
    r = int(np.log2(n))
    if 2 ** r != n:
        raise ValueError("Input vector length must be a power of 2")

    F = np.zeros((n, n), dtype=np.complex128)
    ksi = np.exp(-2.j * np.pi / n)
    for j in range(n):
        for k in range(n):
            val = ksi ** (j * k)
            if abs(val.real) < eps: 
                val = complex(0, val.imag)
            if abs(val.imag) < eps: 
                val = complex(val.real, 0)
            F[j, k] = val

    y = np.dot(F, x)
    return y

def IDFT(y):
    n = len(y)
    F_inv = np.conj(DFT(np.eye(n))) / n
    x = np.dot(F_inv, y)
    return x

# Example usage
x = np.array([1, 2, 3, 4], dtype=np.complex128)
y = DFT(x)
x_reconstructed = IDFT(y)

# Verify against library function
x_reconstructed_library = np.fft.ifft(y)

print("Original x:", x)
print("Reconstructed x using custom IDFT:", x_reconstructed)
print("Reconstructed x using library function:", x_reconstructed_library)
