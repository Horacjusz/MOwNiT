import numpy as np
import time
import matplotlib.pyplot as plt

def gauss_jordan(A, b):
    n = len(A[0])
    h = len(A)
    tab = np.concatenate((A.astype(float), b.reshape(h, 1).astype(float)), axis=1)
    
    for i in range(n - 1):
        # These two lines are responsible for partial pivoting
        max_ind = np.argmax(np.abs(tab[i:, i])) + i
        tab[[i, max_ind]] = tab[[max_ind, i]]
        
        val = tab[i][i]
        if val == 0:
            raise Exception("SINGULAR MATRIX")

        for j in range(i + 1, h):
            tab[j] -= tab[i] * (tab[j][i] / val)

    for i in range(n - 1, -1, -1):
        tab[i] /= tab[i][i]
        for j in range(i - 1, -1, -1):
            tab[j] -= tab[i] * tab[j][i]

    solutions = tab[:, -1][:n]
    return solutions

def testing() :
    random_integers = np.random.randint(500, 2500, size=200)
    sizes = np.sort(random_integers)
    eps = 10**-8

    all_data = []

    my = [0,0]
    numpy = [0,0]

    all_start = time.time()
    for i in range(len(sizes)):
        size = sizes[i]
        A = np.random.rand(size, size)
        b = np.random.rand(size)
        
        print("\nTest nr:",i,"\nSIZE:",size)
        
        
        start_time = time.time()
        x1 = gauss_jordan(A, b)
        end_time = time.time()
        t1 = end_time - start_time
        
        start_time = time.time()
        x2 = np.linalg.solve(A, b)
        end_time = time.time()
        t2 = end_time - start_time
        
        all_data.append([size, t1, t2])
        
        if t1 < t2 :
            my[0] += 1
            numpy[1] += 1
        else :
            numpy[0] += 1
            my[1] += 1
        
        print("TIME IMPLEMENTATION:,",t1,"\nTIME NUMPY:,",t2,"\nFinished")
        for i in range(len(x1)) :
            if abs(x1[i] - x2[i]) > eps : print("WRONG",abs(x1[i] - x2[i]))


    for i in range(2) :
        numpy[i] /= len(sizes)
        my[i] /= len(sizes)

    print(numpy)
    print(my)
                
    all_end = time.time()

    print("\nWHOLE TIME:", all_end - all_start)

    sizes, times_my, times_numpy = zip(*all_data)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_my, label='Exercise implementation', color='blue')
    plt.plot(sizes, times_numpy, label='NumPy', color='red')
    plt.title('Czas działania algorytmu w zależności od rozmiaru macierzy')
    plt.xlabel('Rozmiar macierzy')
    plt.ylabel('Czas [s]')
    plt.legend()
    # plt.savefig('wykres1.png')
    plt.show()

testing()
