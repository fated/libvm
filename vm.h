#ifndef LIBVM_VM_H_
#define LIBVM_VM_H_

#include "utilities.h"
#include "knn.h"

struct Parameter
{
  struct KNNParameter knn_param;
  int num_categories;
};

struct Model
{
  struct Parameter param;
  int l;
  int num_classes;
  int *label;
  int *category;

  double **minD;
  int **minL;
};

struct Model *TrainVM(const struct Problem *train, const struct Parameter *param);

double PredictVM(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper);

int SaveModel(const char *model_file_name, const struct Model *model);
// struct Model *LoadModel(const char *model_file_name);

#endif  // LIBVM_VM_H_
