#ifndef LIBVM_VM_H_
#define LIBVM_VM_H_

#include "utilities.h"
#include "knn.h"

struct Parameter
{
  struct KNNParameter knn_param;
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

double PredictVM(const struct Model *model, const struct Node *x);

#endif  // LIBVM_VM_H_
