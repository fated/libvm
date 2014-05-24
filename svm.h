#ifndef LIBVM_SVM_H_
#define LIBVM_SVM_H_

#include "utilities.h"

enum { C_SVC, NU_SVC };  /* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct SVMParameter
{
  int svm_type;
  int kernel_type;
  int degree;  /* for poly */
  double gamma;  /* for poly/rbf/sigmoid */
  double coef0;  /* for poly/sigmoid */

  /* these are for training only */
  double cache_size; /* in MB */
  double eps;  /* stopping criteria */
  double C;  /* for C_SVC, EPSILON_SVR and NU_SVR */
  int nr_weight;    /* for C_SVC */
  int *weight_label;  /* for C_SVC */
  double* weight;    /* for C_SVC */
  double nu;  /* for NU_SVC, ONE_CLASS, and NU_SVR */
  int shrinking;  /* use the shrinking heuristics */
};

//
// SVMModel
//
struct SVMModel
{
  struct SVMParameter param;  /* parameter */
  int nr_class;    /* number of classes, = 2 in regression/one class svm */
  int l;      /* total #SV */
  struct Node **SV;    /* SVs (SV[l]) */
  double **sv_coef;  /* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
  double *rho;    /* constants in decision functions (rho[k*(k-1)/2]) */
  int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

  /* for classification only */

  int *label;    /* label of each class (label[k]) */
  int *nSV;    /* number of SVs for each class (nSV[k]) */
        /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
  /* XXX */
  int free_sv;    /* 1 if SVMModel is created by LoadSVMModel*/
        /* 0 if SVMModel is created by TrainSVM */
};

struct SVMModel *TrainSVM(const struct Problem *prob, const struct SVMParameter *param);

int SaveSVMModel(const char *model_file_name, const struct SVMModel *model);
struct SVMModel *LoadSVMModel(const char *model_file_name);

int get_svm_type(const struct SVMModel *model);
int get_nr_class(const struct SVMModel *model);
void get_labels(const struct SVMModel *model, int *label);
void get_sv_indices(const struct SVMModel *model, int *sv_indices);
int get_nr_sv(const struct SVMModel *model);

double PredictValues(const struct SVMModel *model, const struct Node *x, double* dec_values);
double PredictSVM(const struct SVMModel *model, const struct Node *x);
double PredictDecisionValues(const struct SVMModel *model, const struct Node *x, double **dec_values);

void FreeSVMModel(struct SVMModel **model_ptr_ptr);
void FreeSVMParam(struct SVMParameter *param);

const char *CheckSVMParameter(const struct Problem *prob, const struct SVMParameter *param);

void svm_set_print_string_function(void (*print_func)(const char *));

#endif  // LIBVM_SVM_H_
