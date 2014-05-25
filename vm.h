#ifndef LIBVM_VM_H_
#define LIBVM_VM_H_

#include "utilities.h"
#include "knn.h"
#include "svm.h"

enum { KNN, SVM };

struct Parameter {
  struct KNNParameter knn_param;
  struct SVMParameter *svm_param;
  int num_categories;
  int save_model;
  int load_model;
  int taxonomy_type;
};

struct Model {
  struct Parameter param;
  struct SVMModel *svm_model;
  int num_ex;
  int num_classes;
  int num_categories;
  int *labels;
  int *categories;

  double **dist_neighbors;
  int **label_neighbors;
};

struct Model *TrainVM(const struct Problem *train, const struct Parameter *param);
double PredictVM(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper);
void OnlinePredict(const struct Problem *prob, const struct Parameter *param, double *predict_labels, int *indices, double *lower_bounds, double *upper_bounds);

int SaveModel(const char *model_file_name, const struct Model *model);
struct Model *LoadModel(const char *model_file_name);
void FreeModel(struct Model *model);
void FreeParam(struct Parameter *param);

const char *CheckParameter(const struct Parameter *param);

#endif  // LIBVM_VM_H_