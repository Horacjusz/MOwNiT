import numpy as np
import time
from random import random

N = 10**7

v = random()*(0.8) + 0.1

print(v)

correct = v*N

print("CORRECT",correct)

base_table = [v]*N

table1 = np.array(base_table, dtype=np.float32)
table2 = np.array(base_table, dtype=np.float64)

def naive(tab) :
    # err_table = []
    summ = 0
    for i in range(len(tab)) :
        v1 = tab[i]
        # if i % 25000 == 0 :
        #     corr = v1*i
        #     err = abs(corr - summ)
        #     err_table.append(err)
        summ += v1
    return summ


def recursive(tab) :
    if len(tab) == 1 :
        return tab[0]
    
    if len(tab)%2 == 1 :
        tab.append(0)
    n_len = len(tab)//2
    
    n_tab = [0]*n_len
    
    for i in range(n_len) :
        n_tab[i] = tab[2*i] + tab[(2*i)+1]
    
    return recursive(n_tab)



# ZAD 2


def kahan(tab) :
    summ = 0
    err = 0

    for val in tab :
        y = val - err
        temp = summ + y
        err = (temp - summ) - y
        summ = temp

    return summ + err


def test_all(tab) :

    naive_start = time.time()
    naive_val = naive(tab)
    naive_end = time.time()
    naive_err = abs(correct - naive_val)
    naive_relative_err = naive_err/correct
    # print(naive_errs)
    print("NAIVE",naive_val,"absolute error:",naive_err,"relative error:",naive_relative_err,"\nTIME:",(naive_end - naive_start),"\n")
    
    
    recursive_start = time.time()
    recursive_val = recursive(tab)
    recursive_end = time.time()
    recursive_err = abs(correct - recursive_val)
    recursive_relative_err = recursive_err/correct
    print("RECURSIVE",recursive_val,"absolute error:",recursive_err,"relative_error",recursive_relative_err,"\nTIME:",(recursive_end - recursive_start),"\n")

    
    print("\nZAD 2\n")
    
    kahan_start = time.time()
    kahan_val = kahan(tab)
    kahan_end = time.time()
    kahan_err = abs(correct - kahan_val)
    kahan_relative_err = kahan_err/correct
    print("KAHAN",kahan_val,"absolute error:",kahan_err,"relative_error",kahan_relative_err,"\nTIME:",(kahan_end - kahan_start),"\n")
    
test_all(table2)

# ZAD 3

# print("\nZAD 3\n")

def dzeta_riemman(s,n,forward = True,precision = 1) :
    if precision == 1 :
        summ = np.float32(0)
        
        if forward :
            for k in range(1,n) :
                val = np.float32(1/(k**s))
                summ += val
        else :
            for k in range(n-1,0,-1) :
                val = np.float32(1/(k**s))
                summ += val
                
        
        return summ
    if precision == 2 :
        summ = np.float64(0)
        
        if forward :
            for k in range(1,n) :
                val = np.float64(1/(k**s))
                summ += val
        else :
            for k in range(n-1,0,-1) :
                val = np.float64(1/(k**s))
                summ += val
                
        
        return summ
    
    
def eta_Dirichlet(s,n,forward = True,precision = 1) :
    
    if precision == 1 :
        summ = np.float32(0)
        
        if forward :
            for k in range(1,n) :
                val = np.float32(((-1)**(k-1))*(1/(k**s)))
                summ += val
        else :
            for k in range(n-1,0,-1) :
                val = np.float32(((-1)**(k-1))*(1/(k**s)))
                summ += val
                
        
        return summ
    
    if precision == 2 :
        summ = np.float64(0)
        
        if forward :
            for k in range(1,n) :
                val = np.float64(((-1)**(k-1))*(1/(k**s)))
                summ += val
        else :
            for k in range(n-1,0,-1) :
                val = np.float64(((-1)**(k-1))*(1/(k**s)))
                summ += val
                
        
        return summ
    
    
s_table = [2,(11/3),5,(72/10),10]
n_table = [50,100,200,500,1000]

error1 = 0
error2 = 0

for s in s_table :
    for n in n_table :
        dzf1 = dzeta_riemman(s,n)
        dzb1 = dzeta_riemman(s,n,False)
        eDf1 = eta_Dirichlet(s,n)
        eDb1 = eta_Dirichlet(s,n,False)
        if dzf1 != dzb1 :     
            print(s,n,"dzeta forward 1",dzf1)
            print(s,n,"dzeta backward 1",dzb1)
            error1 += 1
        if eDb1 != eDf1 :
            print(s,n,"eta forward 1",eDf1)
            print(s,n,"eta backward 1",eDb1)
            error1 += 1
        dzf2 = dzeta_riemman(s,n,precision=2)
        dzb2 = dzeta_riemman(s,n,False,precision=2)
        eDf2 = eta_Dirichlet(s,n,precision=2)
        eDb2 = eta_Dirichlet(s,n,False,precision=2)
        if dzf2 != dzb2 :     
            print(s,n,"dzeta forward 2",dzf2)
            print(s,n,"dzeta backward 2",dzb2)
            error2 += 1
        if eDb2 != eDf2 :
            print(s,n,"eta forward 2",eDf2)
            print(s,n,"eta backward 2",eDb2)
            error2 += 1
        print()
    
print(error1,error2)