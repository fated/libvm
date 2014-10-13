# LibVM -- A Library for Venn Machine

LibVM is a simple, easy-to-use, and efficient software for Venn Machine on classification, which gives label prediction together with it's probabilistic estimations. This library solves Venn prediction in both online and offline mode with _k_-nearest neighbors or support vector machines as the underlying algorithms. This document explains the use of LibVM.

## Table of Contents

* [Installation and Data Format](#installation-and-data-format)
* ["vm-offline" Usage](#vm-offline-usage)
* ["vm-online" Usage](#vm-online-usage)
* ["vm-cv" Usage](#vm-cv-usage)
* [Tips on Practical Use](#tips-on-practical-use)
* [Examples](#examples)
* [Library Usage](#library-usage)
* [Additional Information](#additional-information)
* [Special Thanks](#special-thanks)

## Installation and Data Format

On Unix systems, type `make` to build the `vm-offline`, `vm-online` and `vm-cv` programs. Run them without arguments to show the usage of them.

The format of training and testing data file is:
```
<label> <index1>:<value1> <index2>:<value2> ...
...
...
...
```

Each line contains an instance and is ended by a `'\n'` character (Unix line ending). For classification, `<label>` is an integer indicating the class label (multi-class is supported). For regression, `<label>` is the target value which can be any real number. The pair `<index>:<value>` gives a feature (attribute) value: `<index>` is an integer starting from 1 and `<value>` is the value of the attribute, which could be an integer number or real number. Indices must be in **ASCENDING** order. Labels in the testing file are only used to calculate accuracies and errors. If they are unknown, just fill the first column with any numbers.

A sample classification data set included in this package is `iris_scale` for training and `iris_scale_t` for testing.

Type `vm-offline iris_scale iris_scale_t`, and the program will read the training data and testing data and then output the result into `iris_scale_t_output` file by default. The model file `iris_scale_model` will not be saved by default, however, adding `-s model_file_name` to `[option]` will save the model to `model_file_name`. The output file contains the predicted labels and the lower and upper bounds of probabilities for each predicted label.

## "vm-offline" Usage
```
Usage: vm-offline [options] train_file test_file [output_file]
options:
  -t taxonomy_type : set type of taxonomy (default 0)
    0 -- k-nearest neighbors (KNN)
    1 -- support vector machine with equal length (SVM_EL)
    2 -- support vector machine with equal size (SVM_ES)
    3 -- support vector machine with k-means clustering (SVM_KM)
  -k num_neighbors : set number of neighbors in kNN (default 1)
  -c num_categories : set number of categories for Venn predictor (default 4)
  -s model_file_name : save model
  -l model_file_name : load model
  -p : prefix of options to set parameters for SVM
    -ps svm_type : set type of SVM (default 0)
      0 -- C-SVC    (multi-class classification)
      1 -- nu-SVC   (multi-class classification)
    -pt kernel_type : set type of kernel function (default 2)
      0 -- linear: u'*v
      1 -- polynomial: (gamma*u'*v + coef0)^degree
      2 -- radial basis function: exp(-gamma*|u-v|^2)
      3 -- sigmoid: tanh(gamma*u'*v + coef0)
      4 -- precomputed kernel (kernel values in training_set_file)
    -pd degree : set degree in kernel function (default 3)
    -pg gamma : set gamma in kernel function (default 1/num_features)
    -pr coef0 : set coef0 in kernel function (default 0)
    -pc cost : set the parameter C of C-SVC (default 1)
    -pn nu : set the parameter nu of nu-SVC (default 0.5)
    -pm cachesize : set cache memory size in MB (default 100)
    -pe epsilon : set tolerance of termination criterion (default 0.001)
    -ph shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    -pwi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)
    -pq : quiet mode (no outputs)
```
`train_file` is the data you want to train with.
`test_file` is the data you want to predict.
`vm-offline` will produce outputs in the `output_file` by default.

## "vm-online" Usage
```
Usage: vm-online [options] data_file [output_file]
options:
  -t taxonomy_type : set type of taxonomy (default 0)
    0 -- k-nearest neighbors (KNN)
    1 -- support vector machine with equal length (SVM_EL)
    2 -- support vector machine with equal size (SVM_ES)
    3 -- support vector machine with k-means clustering (SVM_KM)
  -k num_neighbors : set number of neighbors in kNN (default 1)
  -c num_categories : set number of categories for Venn predictor (default 4)
  -p : prefix of options to set parameters for SVM
    -ps svm_type : set type of SVM (default 0)
      0 -- C-SVC    (multi-class classification)
      1 -- nu-SVC   (multi-class classification)
    -pt kernel_type : set type of kernel function (default 2)
      0 -- linear: u'*v
      1 -- polynomial: (gamma*u'*v + coef0)^degree
      2 -- radial basis function: exp(-gamma*|u-v|^2)
      3 -- sigmoid: tanh(gamma*u'*v + coef0)
      4 -- precomputed kernel (kernel values in training_set_file)
    -pd degree : set degree in kernel function (default 3)
    -pg gamma : set gamma in kernel function (default 1/num_features)
    -pr coef0 : set coef0 in kernel function (default 0)
    -pc cost : set the parameter C of C-SVC (default 1)
    -pn nu : set the parameter nu of nu-SVC (default 0.5)
    -pm cachesize : set cache memory size in MB (default 100)
    -pe epsilon : set tolerance of termination criterion (default 0.001)
    -ph shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    -pwi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)
    -pq : quiet mode (no outputs)
```
`data_file` is the data you want to run the online prediction on.
`vm-online` will produce outputs in the `output_file` by default.

## "vm-cv" Usage
```
Usage: vm-cv [options] data_file [output_file]
options:
  -t taxonomy_type : set type of taxonomy (default 0)
    0 -- k-nearest neighbors (KNN)
    1 -- support vector machine with equal length (SVM_EL)
    2 -- support vector machine with equal size (SVM_ES)
    3 -- support vector machine with k-means clustering (SVM_KM)
  -k num_neighbors : set number of neighbors in kNN (default 1)
  -c num_categories : set number of categories for Venn predictor (default 4)
  -v num_folds : set number of folders in cross validation (default 5)
  -p : prefix of options to set parameters for SVM
    -ps svm_type : set type of SVM (default 0)
      0 -- C-SVC    (multi-class classification)
      1 -- nu-SVC   (multi-class classification)
    -pt kernel_type : set type of kernel function (default 2)
      0 -- linear: u'*v
      1 -- polynomial: (gamma*u'*v + coef0)^degree
      2 -- radial basis function: exp(-gamma*|u-v|^2)
      3 -- sigmoid: tanh(gamma*u'*v + coef0)
      4 -- precomputed kernel (kernel values in training_set_file)
    -pd degree : set degree in kernel function (default 3)
    -pg gamma : set gamma in kernel function (default 1/num_features)
    -pr coef0 : set coef0 in kernel function (default 0)
    -pc cost : set the parameter C of C-SVC (default 1)
    -pn nu : set the parameter nu of nu-SVC (default 0.5)
    -pm cachesize : set cache memory size in MB (default 100)
    -pe epsilon : set tolerance of termination criterion (default 0.001)
    -ph shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    -pwi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)
    -pq : quiet mode (no outputs)
```
`data_file` is the data you want to run the cross validation on.
`vm-cv` will produce outputs in the `output_file` by default.

## Tips on Practical Use
* Scale your data. For example, scale each attribute to [0,1] or [-1,+1].
* Try different taxonomies. Some data sets will not achieve good results on some data sets.
* Change parameters for better results especially when you are using SVM related taxonomies.

## Examples
```
> vm-offline -k 3 train_file test_file output_file
```

Train a venn predictor with 3-nearest neighbors as underlying algorithm from `train_file`. Then conduct this classifier to `test_file` and output the results to `output_file`.

```
> vm-offline -t 1 -s model_file train_file test_file
```

Train a venn predictor using support vector machines with equal length intervals as taxonomy from `train_file`. Then conduct this classifier to `test_file` and output the results to the default output file, also the model will be saved to file `model_file`.

```
> vm-online -t 2 data_file
```

Train an online venn predictor classifier using support vector machine with equal size intervals as taxonomy from `data_file`. Then output the results to the default output file.

```
> vm-cv -t 3 -v 10 data_file
```

Do a 10-fold cross validation venn predictor using support vector machine with _k_-means clustering intervals as taxonomy from `data_file`. Then output the results to the default output file.

## Library Usage


## Additional Information
For any questions and comments, please email [c.zhou@cs.rhul.ac.uk](mailto:c.zhou@cs.rhul.ac.uk)

## Acknowledgments
Special thanks to Chih-Chung Chang and Chih-Jen Lin, which are the authors of [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).