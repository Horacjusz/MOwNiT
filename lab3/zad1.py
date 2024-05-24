import mpmath as mp
import numpy as np
import random

# Zdefiniowanie funkcji przy użyciu symboli
def f1(x):
    return mp.cos(x)*mp.cosh(x) - 1

def f2(x):
    return 1/x - mp.tan(x)

def f3(x):
    return 2**(-x) + mp.exp(x) + 2*mp.cos(x) - 6


def bisect(precision, a, b, function, value=0):
    mp.mp.dps = precision
    eps = mp.power(10, -precision)
    a = mp.mpf(a)
    b = mp.mpf(b)
    a += eps
    b -= eps
    value = mp.mpf(value)
    counter = 0

    x = (a + b) / 2
    val = function(x)

    while mp.fabs(val - value) >= eps:
        x = (a + b) / 2
        val = function(x)
        counter += 1

        if ((value - val > eps) and (function((a + x) / 2) - val > eps)) or ((eps < val - value) and (eps < val - function((a + x) / 2))):
            b = x
        elif ((value - val > eps) and (function((b + x) / 2) - val > eps)) or ((eps < val - value) and (eps < val - function((b + x) / 2))):
            a = x
        else:
            break

    return x, counter


if __name__ == "__main__":
    # Tutaj umieść kod, który chcesz wykonać tylko wtedy, gdy plik jest uruchamiany jako skrypt główny

    
    def expected_iterations(a, b, eps):
        return int(mp.ceil(mp.log((b-a)/eps)/mp.log(2)))

    precs = [7, 15, 33] + [random.choice([7, 15, 33]) for _ in range(10)]
    precs.sort()
    
    mp.mp.dps = max(precs)
    eps = mp.power(10,-max(precs))
    intervals = [((3/2)*np.pi, 2*np.pi, f1), (0, np.pi/2, f2), (1, 3, f3)]
    
    summ = mp.mpf(0)

    for a,b,function in intervals:
        for prec in precs:
            print("->", prec)
            eps = np.float64(10**-prec)
            # a, b = interval
            exp_iters = expected_iterations(a, b, eps)
            print(f"Expected: {exp_iters}")
            result = bisect(prec, a, b, function)
            print(f"Iterations: {result[1]}")
            print(f"Start: {a}\nEnd: {b}\nPrecision: {prec}\nResult: {result[0]}\n")
            summ += function(result[0])
            
    summ /= (len(intervals)*len(precs))
    print("avg_value",abs(summ))
