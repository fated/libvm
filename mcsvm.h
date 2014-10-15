#ifndef LIBVM_MCSVM_H_
#define LIBVM_MCSVM_H_

#include "utilities.h"

enum { EXACT, APPROX, BINARY };  // redopt_type
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };  // kernel_type

struct MCSVMParameter {
  int redopt_type;  // reduced optimization type
  int kernel_type;
  int degree;  // for poly
  double gamma;  // for poly/rbf/sigmoid
  double coef0;  // for poly/sigmoid
  double beta;
  double cache_size; // in Mb
  double epsilon;
  double epsilon0;
  double delta;
  // enum RedOptType redopt_type;
};

struct MCSVMModel {
  struct MCSVMParameter param;
  int num_ex;
  int num_classes;  // number of classes (k)
  int total_sv;  // total #SV
  struct Node **svs;  // SVs (SV[total_sv])
  double **sv_coef;  // coefficients for SVs in decision functions (sv_coef[k-1][total_sv])
  double *rho;  // constants in decision functions (rho[k*(k-1)/2])
  int *sv_indices;  // sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set
  int *labels;  // label of each class (label[k])
  int *num_svs;  // number of SVs for each class (nSV[k])
                 // nSV[0] + nSV[1] + ... + nSV[k-1] = total_sv
  int free_sv;  // 1 if SVMModel is created by LoadSVMModel
                // 0 if SVMModel is created by TrainSVM
};

MCSVMModel *TrainMCSVM(const struct Problem *prob, const struct MCSVMParameter *param);
double PredictMCSVM(const struct MCSVMModel *model, const struct Node *x);

int SaveMCSVMModel(std::ofstream &model_file, const struct MCSVMModel *model);
MCSVMModel *LoadMCSVMModel(std::ifstream &model_file);
void FreeMCSVMModel(struct MCSVMModel **model);

void FreeMCSVMParam(struct MCSVMParameter *param);
void InitMCSVMParam(struct MCSVMParameter *param);
const char *CheckMCSVMParameter(const struct MCSVMParameter *param);

#endif  // LIBVM_MCSVM_H_