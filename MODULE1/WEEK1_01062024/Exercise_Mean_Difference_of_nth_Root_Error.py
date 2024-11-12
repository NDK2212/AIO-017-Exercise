from math import *
def md_nre_single_sample(y,y_hat,n,p):
    return ((y**(1/n)) - (y_hat**(1/n)))**p

if __name__ == "__main__":
    try:
        y = float(input("Enter y: "))
        y_hat = float(input("Enter y_hat: "))
        n = int(input("Enter the nth root: "))
        p = int(input("Enter the loss_level: "))
    except ValueError:
        print("Your enter is invalid.")
    print(md_nre_single_sample(y,y_hat,n,p))