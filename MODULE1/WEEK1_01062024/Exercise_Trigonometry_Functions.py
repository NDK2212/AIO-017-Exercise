from math import *
def factorrial(x):
    if x == 1:
        return 1
    else:
        return x * factorial(x-1)
    
def sin_calculation(x,n):
    result = 0
    for i in range(0,n+1):
        result+= ((-1)**i)*((x**((2*i)+1))/factorial((2*i) + 1))
    return result

def cos_calculation(x,n):
    result = 0
    for i in range(0,n+1):
        result+= ((-1)**i)*((x**(2*i))/factorial(2*i))
    return result

def sinh_calculation(x,n):
    result = 0
    for i in range(0,n+1):
        result+=((x**((2*i)+1))/factorial((2*i) + 1))
    return result

def cosh_calculation(x,n):
    result = 0
    for i in range(0,n+1):
        result+= ((x**(2*i))/factorial(2*i))
    return result

if __name__ == "__main__":
    x = float(input("Enter the value of x:"))
    n = input("Enter the number of iterations:")
    if not n.isnumeric():
        print("Number of iteration must be a integer.")
        quit()
    else:
        n = int(n)
    if  n <= 0:
        print("Number of iterations must be greater than 0.")
        quit()
    print(f"sin(x = {x}, n = {n}): {sin_calculation(x,n)}")  
    print(f"cos(x = {x}, n = {n}): {cos_calculation(x,n)}")  
    print(f"sinh(x = {x}, n = {n}): {sinh_calculation(x,n)}")  
    print(f"cosh(x = {x}, n = {n}): {cosh_calculation(x,n)}")  

    