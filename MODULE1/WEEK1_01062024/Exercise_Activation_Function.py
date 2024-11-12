from math import *
def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def sigmoid(n):
    return 1/(1+exp(-n)) # exp(n) == e**n

def relu(n):
    if n>0:
        return n
    else:
        return 0
    
def elu(n):
    a = 0.01
    if n > 0:
        return n
    else:
        return a * (exp(n) - 1)

if __name__ == "__main__":
    x = input("Input x = ")
    if not is_number(x):
        print("x must be a number")
        quit()
    else:
        x = float(x)
    activation_function = input("Input activation Function ( sigmoid | relu | elu ) :")
    list_of_activation = ['sigmoid','relu','elu']
    if activation_function not in list_of_activation:
        print(f"{activation_function} is not supported")
        quit()
    if activation_function == list_of_activation[0]:
        result = sigmoid(x)
        print(f"sigmoid: f({x}) = {result}")
    elif activation_function == list_of_activation[1]:
        result = relu(x)
        print(f"relu: f({x}) = {result}")
    elif activation_function == list_of_activation[2]:
        result = elu(x)
        print(f"elu: f({x}) = {result}")
        
        