## Requirements
The project requires one library.
- [TFHE](https://github.com/tfhe/tfhe)

## HOW TO COMPILE  
#### CLASSIFICATION MNIST
``` g++ classification_mnist.cpp -Ofast -o classification_mnist -ltfhe-spqlios-fma -fopenmp```  
#### CLASSIFICATION BREAST CANCER WISCONSIN
``` g++ classification_breast.cpp -Ofast -o classification_breast -ltfhe-spqlios-fma -fopenmp```  

Note the libraries are assumed to be installed in their default location (/usr/local/lib).    

## HOW TO RUN  
#### CLASSIFICATION MNIST
``` ./classification_mnist [mode] [end] [start] [threads]```
#### CLASSIFICATION BREAST CANCER WISCONSIN
``` ./classification_breast [end] [start] [threads]```    
where:
- [mode] **bool**:  
    0 for 30 neurons in the hidden layer  
    1 for 300   
    - default = 0
- [end] **int**: ending entry
    - default = 1
- [start] **int**: starting entry
    - default = 0
- [threads] **int**: number of processes
    - default = 1
