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
* [Acknowledgments](#acknowledgments)

## Installation and Data Format[↩](#table-of-contents)

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

## "vm-offline" Usage[↩](#table-of-contents)
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

## "vm-online" Usage[↩](#table-of-contents)
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

## "vm-cv" Usage[↩](#table-of-contents)
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

## Tips on Practical Use[↩](#table-of-contents)
* Scale your data. For example, scale each attribute to [0,1] or [-1,+1].
* Try different taxonomies. Some data sets will not achieve good results on some data sets.
* Change parameters for better results especially when you are using SVM related taxonomies.

## Examples[↩](#table-of-contents)
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

## Library Usage[↩](#table-of-contents)
All functions and structures are declared in different header files. There are 5 parts in this library, which are **utilities**, **knn**, **svm**, **vm** and the other driver programs.

### `utilities.h` and `utilities.cpp`
The structure `Problem` for storing the data sets (including the structure `Node` for storing the attributes pair of index and value) and all the constant variables are declared in `utilities.h`.

In this file, some utilizable function templates or functions are also declared.

* `T FindMostFrequent(T *array, int size)`  
  This function is used to find the most frequent category in _k_NN taxonomy.
* `static inline void clone(T *&dest, S *src, int size)`  
  This static function is used to clone an array from `src` to `dest`.
* `void QuickSortIndex(T array[], size_t index[], size_t left, size_t right)`  
  This function is used to quicksort an array and preserve the original indices.
* `Problem *ReadProblem(const char *file_name)`  
  This function is used to read in a data set from a file named `file_name`.
* `void FreeProblem(struct Problem *problem)`  
  This function is used to free a problem stored in the memory.
* `void GroupClasses(const Problem *prob, int *num_classes_ret, int **labels_ret, int **start_ret, int **count_ret, int *perm)`  
  This function is used in Cross Validation and other predictions using SVM related taxonomies. This function will group the examples with same label together. The last 5 parameters are using to return corresponding values. `num_classes_ret` is used to store the number of classes in the problem. `labels_ret` is an array used to store the actual label in the order of appearance. `start_ret` is an array used to store the starting index of each group of examples. `count_ret` is an array used to store the count number of each group of examples. `perm` is an array used to store the permutation of the permuted index of the problem.

### `knn.h` and `knn.cpp`
The structure `KNNParameter` for storing the _k_NN related parameters and the structure `KNNModel` for storing the _k_NN related model are declared in `knn.h`.

In this file, some utilizable function templates or functions are also declared.

* `static inline void InsertLabel(T *labels, T label, int num_neighbors, int index)`  
  This static function will insert `label` into the `index`-th location of the array `labels` of which the size is `num_neighbors`.
* `KNNModel *TrainKNN(const struct Problem *prob, const struct KNNParameter *param)`  
  This function is used to train a _k_NN model from a problem `prob` and the parameter `param`, it will return a model of the structure `KNNModel`.
* `double PredictKNN(struct Problem *train, struct Node *x, const int num_neighbors)`  
  This function is used to predict the label for object `x` using _k_NN classifier.
* `double CalcDist(const struct Node *x1, const struct Node *x2)`  
  This function is used to calculate the distance between two objects `x1` and `x2`, which will be used in _k_NN.
* `int CompareDist(double *neighbors, double dist, int num_neighbors)`  
  This function is used to compare a distance `dist` with the nearest neighbors' distances stored in an array `neighbors`, it will return the position of `dist`, if it is greater than all the distances in `neighbors`, it gives `num_neighbors`.
* `int SaveKNNModel(std::ofstream &model_file, const struct KNNModel *model)`
* `KNNModel *LoadKNNModel(std::ifstream &model_file)`
* `void FreeKNNModel(struct KNNModel *model)`  
  These three functions are used to manipulate the _k_NN model file, including "save to file", "load from file" and "free the model".
* `void FreeKNNParam(struct KNNParameter *param)`
* `void InitKNNParam(struct KNNParameter *param)`
* `const char *CheckKNNParameter(const struct KNNParameter *param)`  
  These three functions are used to manipulate the _k_NN parameter file, including "free the param", "initial the param" and "check the param".

### `svm.h` and `svm.cpp`
The structure `SVMParameter` for storing the SVM related parameters and the structure `SVMModel` for storing the SVM related model are declared in `svm.h`.

In this file, some utilizable function templates or functions are also declared.

* `SVMModel *TrainSVM(const struct Problem *prob, const struct SVMParameter *param)`  
  This function is used to train a SVM model from a problem `prob` and the parameter `param`, it will return a model of the structure `SVMModel`.
* `double PredictValues(const struct SVMModel *model, const struct Node *x, double* decision_values)`  
  This function is used to predict the label for object `x` using SVM classifier.
* `double PredictSVM(const struct SVMModel *model, const struct Node *x)`  
  This function is an interface for `PredictValues()` to predict label.
* `double PredictDecisionValues(const struct SVMModel *model, const struct Node *x, double **decision_values)`  
  This function is an interface for `PredictValues()` to predict label and get `decision_values`.
* `int SaveSVMModel(std::ofstream &model_file, const struct SVMModel *model)`
* `SVMModel *LoadSVMModel(std::ifstream &model_file)`
* `void FreeSVMModel(struct SVMModel **model)`  
  These three functions are used to manipulate the SVM model file, including "save to file", "load from file" and "free the model".
* `void FreeSVMParam(struct SVMParameter *param)`
* `void InitSVMParam(struct SVMParameter *param)`
* `const char *CheckSVMParameter(const struct SVMParameter *param)`  
  These three functions are used to manipulate the SVM parameter file, including "free the param", "initial the param" and "check the param".
* `void SetPrintNull()`
* `void SetPrintCout()`  
  These two functions are used to set the output destination. `SetPrintNull()` will print the output to nowhere (except the warning and error messages and the final results). `SetPrintCout()` will print the output to the standard output stream.

### `vm.h` and `vm.cpp`
The structure `Parameter` for storing the Venn Machine related parameters and the structure `Model` for storing the Venn Machine related model are declared in `vm.h`. You need to #include "vm.h" in your C/C++ source files and
link your program with `vm.cpp`. You can see `vm-offline.cpp`,
`vm-online.cpp` and `vm-cv.cpp` for examples showing how to use them.

In this file, some utilizable function templates or functions are also declared.

* `Model *TrainVM(const struct Problem *train, const struct Parameter *param)`  
  This function is used to train a venn predictor from the problem `train` and the parameter `param`.
* `double PredictVM(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper, double **avg_prob)`  
  This function is used to predict a new object `x` from the problem `train` and the `model`. It will return the predicted label, `lower` for lower bound of the probability, `upper` for upper bound and `avg_prob` for calculate performance measures are also returned.
* `void CrossValidation(const struct Problem *prob, const struct Parameter *param, double *predict_labels, double *lower_bounds, double *upper_bounds, double *brier, double *logloss)`  
  This function is used to do a cross validation on the problem `prob` and the parameter `param`. The other 5 parameters are used to return the corresponding values.
* `void OnlinePredict(const struct Problem *prob, const struct Parameter *param, double *predict_labels, int *indices, double *lower_bounds, double *upper_bounds, double *brier, double *logloss)`  
  This function is used to do a online prediction on the problem `prob` and the parameter `param`. The other 6 parameters are used to return the corresponding values.
* `int SaveModel(const char *model_file_name, const struct Model *model)`
* `Model *LoadModel(const char *model_file_name)`
* `void FreeModel(struct Model *model)`  
  These three functions are used to manipulate the model file, including "save to file", "load from file" and "free the model".
* `void FreeParam(struct Parameter *param)`
* `const char *CheckParameter(const struct Parameter *param)`  
  These two functions are used to manipulate the parameter file, including "free the param" and "check the param".

### `vm-offline.cpp`, `vm-online.cpp` and `vm-cv.cpp`
These three files are the driver programs for LibVM. `vm-offline.cpp` is for training and testing data sets in offline setting. `vm-online.cpp` is for doing online prediction on data sets. `vm-cv.cpp` is for doing cross validation on data sets.

The structure of these files are similar. In these programs, the command-line inputs will be parsed, the data sets will be read into the memory, the train and predict process will be called, the performance measure process will be carried out and finally the memories it claimed will be cleaned up. It includes the following functions.

* `void ExitWithHelp()`  
  This function is used to print out the usage of the executable file.
* `void ParseCommandLine(int argc, char *argv[], ...)`  
  This function is used to parse the options from the command-line input, and return the values like file names to the other parameters which is represented by `...`.

## Additional Information[↩](#table-of-contents)
For any questions and comments, please email [c.zhou@cs.rhul.ac.uk](mailto:c.zhou@cs.rhul.ac.uk)

## Acknowledgments[↩](#table-of-contents)
Special thanks to Chih-Chung Chang and Chih-Jen Lin, which are the authors of [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).
