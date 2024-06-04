from math import *
def classification_model_evaluation(tp,fp,fn):
    if not (isinstance(tp,int) and isinstance(fp,int) and isinstance(fn,int)):
        print("TP and FP and FN must of float type.")
        return
    if tp <= 0 or fp<=0 or fn<=0:
        print("TP and FP and FN must be greater than 0")
        return
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    F1_Score = 2*((precision*recall)/(precision+recall))
    print(f"precision is {precision}")
    print(f"recall is {recall}")
    print(f"f1-score is {F1_Score}")
    
if __name__ == "__main__":
    classification_model_evaluation(tp=-10,fp=4,fn=5)
# this source code requires the input being entered right into the code (as the exercise tells me to)
# if input from user is required => use input() function and check the data type using try and except  
    


    