# SOLUTION TO PART 2

import sys          #For getting the arguments from the commandline
import numpy as np  #For storing the contents of the CSV as a Numpy Array
import os  #For removing the output_file if it already exists

from GradientDescent import GradientDescent #The Logic is implemented in GradientDescent


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
        input = np.genfromtxt(input_file, delimiter=',')
        num_features = input.shape[1] #Number of features 
        X = input[:, 0:(num_features-1)] 
        Y = input[:, -1]

        output_file = sys.argv[2]
        
        try:
            os.remove(output_file)
        except OSError:
            pass
        
        GradientDescent(X, Y, output_file)

    else:
        print ("INVALID NUMBER OF INPUTS") 
