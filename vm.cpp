#include "utilities.h"
#include "knn.h"
#include "vm.h"
#include <iostream>
#include <cmath>

#define INF HUGE_VAL
#define TAU 1e-12

struct Model *TrainVM(const struct Problem *train, const struct Parameter *param)
{
  Model *model = new Model;
  model->param = *param;

  int l = train->l;
  int num_classes = 0;
  int num_neighbors = param->knn_param.num_neighbors;
  int *label = NULL;

  std::vector<int> unique_label;
  for (int i = 0; i < l; ++i) {
    int this_label = (int) train->y[i];
    std::size_t j;
    for (j = 0; j < num_classes; ++j) {
      if (this_label == unique_label[j]) {
        break;
      }
    }
    if (j == num_classes) {
      unique_label.push_back(this_label);
      ++num_classes;
    }
  }
  label = new int[num_classes];
  for (std::size_t i = 0; i < unique_label.size(); ++i) {
    label[i] = unique_label[i];
  }
  std::vector<int>(unique_label).swap(unique_label);

  if (num_classes == 1)
    std::cout << "WARNING: training set only has one class. See README for details." << std::endl;

  int *category = new int[l];
  double **minD = new double*[l];
  int **minL = new int*[l];

  for (int i = 0; i < l; ++i) {
    minD[i] = new double[num_neighbors];
    minL[i] = new int[num_neighbors];
    for (int j = 0; j < num_neighbors; ++j) {
      minD[i][j] = INF;
      minL[i][j] = -1;
    }
    category[i] = -1;
  }

  for (int i = 0; i < l-1; ++i) {
    for (int j = i+1; j < l; ++j) {
      double dist = CalcDist(train->x[i], train->x[j]);

      int idx = CompareDist(minD[i], dist, num_neighbors);
      if (idx < num_neighbors) {
        int k;
        for (k = 0; k < num_classes; ++k) {
          if (label[k] == train->y[j]) {
            break;
          }
        }
        for (int m = num_neighbors-1; m > idx; --m)
            minL[j][m] = minL[j][m-1];
          minL[j][idx] = k;
      }
      idx = CompareDist(minD[j], dist, num_neighbors);
      if (idx < num_neighbors) {
        int k;
        for (k = 0; k < num_classes; ++k) {
          if (label[k] == train->y[j]) {
            break;
          }
        }
        for (int m = num_neighbors-1; m > idx; --m)
            minL[j][m] = minL[j][m-1];
          minL[j][idx] = k;
      }
    }
  }

  for (int i = 0; i < l; ++i) {
    int *voting = new int[num_classes];
    for (int j = 0; j < num_classes; ++j) {
      voting[j] = 0;
    }

    for (int j = 0; j < num_neighbors; ++j) {
      voting[minL[i][j]]++;
    }
    int max_vot = voting[0];
    category[i] = 0;
    for (int j = 1; j < num_classes; ++j)
    {
      if (voting[j] > max_vot)
      {
        max_vot = voting[j];
        category[i] = j;
      }
    }
    delete[] voting;
  }

  model->num_classes = num_classes;
  model->l = l;
  model->label = label;
  model->category = category;
  model->minD = minD;
  model->minL = minL;

  return model;
}

double PredictVM(const struct Model *model, const struct Node *x)
{
  return 0;
}