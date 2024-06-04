from math import *
import random
def MAE_calculation(pre,tar):
    return abs(pre-tar)
def MSE_calculation(pre,tar):
    return abs(pre-tar)**2
def RMSE_calculation(pre,tar):
    return abs(pre-tar)**2

if __name__ == "__main__":
    sample_size = input("Input number of samples ( integer number ) which are generated :")
    if not sample_size.isnumeric():
        print("number of samples must be an integer number")
        quit()
    else:
        sample_size = int(sample_size)
    
    loss_name = input("Input loss name (MAE | MSE | RMSE): ")
    list_of_sample =[]
    for sample_index in range(0,sample_size):
        predict = random.uniform(0,10)
        target = random.uniform(0,10)
        if loss_name == "MAE":
            loss = MAE_calculation(predict,target)
            list_of_sample.append(("MAE",predict,target,loss))
        elif loss_name == "MSE":
            loss = MSE_calculation(predict,target)
            list_of_sample.append(("MSE",predict,target,loss))
        elif loss_name == "RMSE":
            loss = RMSE_calculation(predict,target)
            list_of_sample.append(("RMSE",predict,target,loss))
    print(len(list_of_sample))
    sum_of_loss = 0
    for sample_index in range(0,sample_size):
        sum_of_loss+=list_of_sample[sample_index][3]
        print(f"loss name: {list_of_sample[sample_index][0]}, sample: {sample_index}, pred: {list_of_sample[sample_index][1]}, target: {list_of_sample[sample_index][2]}, loss: {list_of_sample[sample_index][3]}")
    if loss_name == "MAE":
        print(f"Final MAE: {sum_of_loss/sample_size}")
    elif loss_name == "MSE":
        print(f"Final MSE: {sum_of_loss/sample_size}")
    elif loss_name == "RMSE":
        print(f"Final RMSE: {sqrt(sum_of_loss/sample_size)}")
        
        
        