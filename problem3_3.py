# SOLUTION TO PART 3

import sys          #For getting the arguments from the commandline
import pandas as pd  #For storing the contents of the CSV in a DataFrame
import os  #For removing the output_file if it already exists

from SVMModels import SVM #The Logic is implemented in SVM


#THE MAIN FUNCTION GOES HERE
if __name__ == "__main__":
    '''
    The function takes arguments from commandline
    Argument 1 - Python file name 
    Argument 2 - Input CSV file name
                 These inputs are stored in arrays X and Y, where X contains the data and Y the labels
    Argument 3 - Output CSV file name 
    '''
    if len(sys.argv)==3:
        input_file = sys.argv[1]
        input = pd.read_csv(input_file, header=0)
        input = input.as_matrix()
        num_features = input.shape[1] #Number of features 
        X = input[:, 0:(num_features-1)] 
        Y = input[:, -1]

        output_file = sys.argv[2]
        
        try:
            os.remove(output_file)
        except OSError:
            pass
        
        for i in range(1, 8):
        	SVM(X=X, Y=Y, type=i, output_file=output_file)

    else:
        print ("INVALID NUMBER OF INPUTS") 
