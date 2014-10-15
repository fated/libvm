#include "mcsvm.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

MCSVMModel *TrainMCSVM(const struct Problem *prob, const struct MCSVMParameter *param) {

}

double PredictMCSVM(const struct MCSVMModel *model, const struct Node *x) {

}

int SaveMCSVMModel(std::ofstream &model_file, const struct MCSVMModel *model) {

}

MCSVMModel *LoadMCSVMModel(std::ifstream &model_file) {

}

void FreeMCSVMModel(struct MCSVMModel **model) {

}

void FreeMCSVMParam(struct MCSVMParameter *param) {

}

void InitMCSVMParam(struct MCSVMParameter *param) {

}

const char *CheckMCSVMParameter(const struct MCSVMParameter *param) {

}