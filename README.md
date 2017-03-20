# Neural network
Neural Net w/o using MATLAB data science toolbox  

Neural network has been implemented for whitespace delimited data with last collumn representing class  
This is a feed forward neural network with backpropagation for updating weights  

samples of accepted datasets have been included in this folder  

The first argument, <training_file>, is the path name of the training file.  
The second argument, <test_file>, is the path name of the test file.  
The third argument, <layers>, specifies how many layers to use. Note that the input layer is layer 1, so the number of layers cannot be smaller than 2.  
The fourth argument,<units_per_layer>, specifies how many perceptrons to place at each hidden layer. This number excludes the bias input.   
The fifth argument, <rounds>, is the number of training rounds that the algorithm will use as stopping criteria.  


place the contents of the folder in the mounted MATLAB folder  
The MATLAB script has been implemented as a function  

TO RUN THE CODE  
IN THE COMMAND WINDOW INSIDE MATLAB  
TYPE AND THEN PRESS ENTER  
neural_network('training_file' ,'test_file' ,layers,units_per_layer,rounds)  

example  
neural_network('pendigits_training.txt', 'pendigits_test.txt', 2, 0, 50)  

neural_network('pendigits_training.txt', 'pendigits_test.txt', 3, 20, 20)  


DO NOT RUN AS  
neural_network <training_file> <test_file> <layers> <units_per_layer> <rounds>  

example of incorrect call  
neural_network pendigits_training.txt pendigits_test.txt 3 20 20  
