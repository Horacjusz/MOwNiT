import mpmath as mp
import numpy as np
import time
import random
from zad1 import f1, f2, f3, bisect


def newton(precision, initial_guess, function, max_iterations=1000, epsilon=1e-10):
    mp.mp.dps = precision
    x = mp.mpf(initial_guess)
    eps = mp.power(10, -precision)
    counter = 0
    der = derivative(function)  # Funkcja do obliczania pochodnej

    while counter < max_iterations:
        f_x = function(x)
        if mp.fabs(f_x) < epsilon:
            return x, counter  # Zbiegło w obrębie epsilon

        f_prime_x = der(x)
        if mp.fabs(f_prime_x) < epsilon:
            raise ValueError("Derivative too small.")

        x_new = x - f_x / f_prime_x

        if mp.fabs(x_new - x) < eps:
            return x_new, counter  # Zbiegło z zadaną precyzją

        x = x_new
        counter += 1

    raise ValueError("Max iterations reached.")  # Nie zbiegło w max_iterations iteracji


# Obliczenie pochodnych funkcji
def derivative(function):
    return lambda x: mp.diff(function, x)


if __name__ == "__main__":
    # Przykładowe przedziały dla testów
    
    precs = [7, 15, 33] + [random.choice([7, 15, 33]) for _ in range(10)]
    precs.sort()
    mp.mp.dps = max(precs)
    eps = mp.power(10,-max(precs))
    intervals = [((3/2)*np.pi, 2*np.pi, f1), (0, np.pi/2, f2), (1, 3, f3)]
    

    newton_wins = 0
    bisect_wins = 0
    equal = 0
    all_tests = 0

    for a, b, function in intervals:
        for prec in precs:
            print(f"Precision: {prec}")
            newton_start = time.time()
            result = newton(prec, (a+b)/2, function)
            newton_end = time.time()
            newton_time = newton_end - newton_start

            print(f"Iterations: {result[1]}")
            print(f"Result: {result[0]}")

            bisect_start = time.time()
            result_bisect = bisect(prec, a, b, function)
            bisect_end = time.time()
            bisect_time = bisect_end - bisect_start

            print("NEWTON TIME", newton_time)
            print("BISECT TIME", bisect_time)
            if newton_time < bisect_time:
                print("NEWTON WINS")
                newton_wins += 1
            elif newton_time > bisect_time:
                print("BISECT WINS")
                bisect_wins += 1
            else:
                print("EQUAL")
                equal += 1
            all_tests += 1
            print("\n")

    print("Newton:", newton_wins/all_tests)
    print("Bisect:", bisect_wins/all_tests)
    print("Equal:", equal/all_tests)
