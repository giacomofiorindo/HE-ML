# Python Implementation
This part of the project is dedicated to the construction of a framework to
easily build and train muti-class feed-forward neural networks. The trained
models can be found in the [resources folder](../../resources/). They will be used for the C++
part of the project.

## Requirements
The requirements can be found [here](requirements.txt). 

NOTE: Keras is required only if the datased is missing from the resources folder.
 
## Usage
Compare the original model with the discretized versions for the [Breast Cancer Wisconsin dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
    
    python compare_breast.py
Compare the original model with the discretized versions  for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
    
    python compare_mnist.py

## Extra programs
#### Training
Train the model for the Breast Cancer Wisconsin dataset
    
    python train_breast.py
Train the model for the MNIST dataset

    python train_mnist.py
#### Resources

Generate resources for the Breast Cancer Wisconsin dataset
    
    python train_breast.py
Generate resources for the MNIST dataset
    
    python train_mnist.py