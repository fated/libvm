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
  int *labels;
  int *categories;

  double **dist_neighbors;
  int **label_neighbors;
};

struct Model *TrainVM(const struct Problem *train, const struct Parameter *param);

double PredictVM(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper);

int SaveModel(const char *model_file_name, const struct Model *model);
// struct Model *LoadModel(const char *model_file_name);
void FreeModel(struct Model *model);

#endif  // LIBVM_VM_H_
