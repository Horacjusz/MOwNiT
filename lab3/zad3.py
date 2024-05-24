import mpmath as mp
import numpy as np
import time
import random
from zad1 import f1, f2, f3, bisect
from zad2 import newton

def secant(prec, a, b, f, max_iterations=100, value = 0):
    mp.mp.dps = prec  # Set precision
    eps = mp.power(10, -prec)
    
    # Convert initial guesses to mpf objects
    x0 = mp.mpf(a)
    x1 = mp.mpf(b)
    x0 += eps
    x1 -= eps
     
    for i in range(max_iterations):
        
        
        # Calculate next approximation
        try :
            x_next = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
        except :
            return None,None
        
        # Check for convergence
        if mp.fabs(function(x_next) - value) < eps:
            return x_next, i + 1  # Return the root and number of iterations
        
        
        # Update guesses
        x0 = x1
        x1 = x_next
    
    # print("ITERAQTIONS")
    # If max_iterations reached without convergence, return the last approximation
    return x_next, max_iterations

if __name__ == "__main__":
    # Przykładowe przedziały dla testów
    precs = [7, 15, 33] + [random.choice([7, 15, 33]) for _ in range(10)]
    precs.sort()
    mp.mp.dps = max(precs)
    
    eps = mp.power(10,-max(precs))
    
    intervals = [((3/2)*np.pi, 2*np.pi, f1), (0, np.pi/2, f2), (1, 3, f3)]
    
    summ_secant = mp.mpf(0)
    summ_newton = mp.mpf(0)
    summ_bisect = mp.mpf(0)
    
    newton_wins = [0,0,0]
    bisect_wins = [0,0,0]
    secant_wins = [0,0,0]
    equal = 0
    all_tests = 0

    for a, b, function in intervals:
        for prec in precs:
            
            secant_start = time.time()
            result = secant(prec,a,b,function)
            secant_end = time.time()
            secant_time = secant_end - secant_start
            
            if result[0] is None :
                continue
            
            
            print(f"Precision: {prec}")
            print(f"Iterations: {result[1]}")
            print(f"Result: {result[0]}")

            newton_start = time.time()
            result_newton = newton(prec, (a+b)/2, function)
            newton_end = time.time()
            newton_time = newton_end - newton_start
            
            bisect_start = time.time()
            result_bisect = bisect(prec, a, b, function)
            bisect_end = time.time()
            bisect_time = bisect_end - bisect_start
            
            print("SECANT TIME",secant_time)
            print("NEWTON TIME",newton_time)
            print("BISECT TIME",bisect_time)
            
            summ_secant += function(result[0])
            summ_newton += function(result_newton[0])
            summ_bisect += function(result_bisect[0])
            
            
            if newton_time < bisect_time < secant_time :
                newton_wins[0] += 1
                bisect_wins[1] += 1
                secant_wins[2] += 1
            elif newton_time < secant_time < bisect_time :
                newton_wins[0] += 1
                secant_wins[1] += 1
                bisect_wins[2] += 1
            elif secant_time < newton_time < bisect_time :
                secant_wins[0] += 1
                newton_wins[1] += 1
                bisect_wins[2] += 1
            elif secant_time < bisect_time < newton_time :
                secant_wins[0] += 1
                bisect_wins[1] += 1
                newton_wins[2] += 1
            elif bisect_time < secant_time < newton_time :
                bisect_wins[0] += 1
                secant_wins[1] += 1
                newton_wins[2] += 1
            elif bisect_time < newton_time < secant_time :
                bisect_wins[0] += 1
                newton_wins[1] += 1
                secant_wins[2] += 1
            all_tests += 1
            print("\n")
            
    summ_secant /= all_tests
    summ_newton /= all_tests
    summ_bisect /= all_tests
    print("avg secant",abs(summ_secant))
    print("avg newton",abs(summ_newton))
    print("avg bisect",abs(summ_newton))
    print()
    print(secant_wins,"SECANT")
    print(newton_wins,"NEWTON")
    print(bisect_wins,"BISECT")
