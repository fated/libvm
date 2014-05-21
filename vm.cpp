#include "utilities.h"
#include "knn.h"
#include "vm.h"
#include <iostream>
#include <cmath>

template <typename T, typename S> static inline void clone(T *&dest, S *src, int size)
{
  dest = new T[size];
  if (sizeof(T) < sizeof(S))
    std::cout << "WARNING: destination type is smaller than source type, data will be truncated." << std::endl;
  std::copy(src, src+size, dest);
}

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
        InsertLabel(minL[i], k, num_neighbors, idx);
      }
      idx = CompareDist(minD[j], dist, num_neighbors);
      if (idx < num_neighbors) {
        int k;
        for (k = 0; k < num_classes; ++k) {
          if (label[k] == train->y[i]) {
            break;
          }
        }
        InsertLabel(minL[j], k, num_neighbors, idx);
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

double PredictVM(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper)
{
  const Parameter& param = model->param;
  int l = model->l;
  int num_classes = model->num_classes;
  int num_neighbors = param.knn_param.num_neighbors;
  // int num_categories = num_classes;
  int *label = model->label;
  double predict_label;

  int **f_matrix = new int*[num_classes];
  double y;

  for (int i = 0; i < num_classes; ++i) {
    y = label[i];

    int *category = new int[l+1];
    double **minD = new double*[l+1];
    int **minL = new int*[l+1];
    f_matrix[i] = new int[num_classes];
    for (int j = 0; j < num_classes; ++j) {
      f_matrix[i][j] = 0;
    }

    for (int j = 0; j < l; ++j) {
      clone(minD[j], model->minD[j], num_neighbors);
      clone(minL[j], model->minL[j], num_neighbors);
      category[j] = model->category[j];
    }
    minD[l] = new double[num_neighbors];
    minL[l] = new int[num_neighbors];
    for (int j = 0; j < num_neighbors; ++j) {
      minD[l][j] = INF;
      minL[l][j] = -1;
    }
    category[l] = -1;

    for (int j = 0; j < l; ++j) {
      double dist = CalcDist(train->x[j], x);

      int idx = CompareDist(minD[j], dist, num_neighbors);
      if (idx < num_neighbors) {
        InsertLabel(minL[j], i, num_neighbors, idx);
      }
      idx = CompareDist(minD[l], dist, num_neighbors);
      if (idx < num_neighbors) {
        int k;
        for (k = 0; k < num_classes; ++k)
          if (label[k] == train->y[j])
            break;
        InsertLabel(minL[l], k, num_neighbors, idx);
      }
    }

    for (int j = 0; j < l+1; ++j) {
      int *voting = new int[num_classes];
      for (int k = 0; k < num_classes; ++k) {
        voting[k] = 0;
      }

      for (int k = 0; k < num_neighbors; ++k) {
        voting[minL[j][k]]++;
      }
      int max_vot = voting[0];
      category[j] = 0;
      for (int k = 1; k < num_classes; ++k) {
        if (voting[k] > max_vot) {
          max_vot = voting[k];
          category[j] = k;
        }
      }
      delete[] voting;
    }

    for (int j = 0; j < l; ++j) {
      if (category[j] == category[l]) {
        for (int k = 0; k < num_classes; ++k) {
          if (label[k] == train->y[j]) {
            f_matrix[i][k]++;
            break;
          }
        }
      }
    }
    f_matrix[i][i]++;

    for (int j = 0; j < l+1; ++j) {
      delete[] minD[j];
      delete[] minL[j];
    }

    delete[] minD;
    delete[] minL;
    delete[] category;
  }

  double **matrix = new double*[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    matrix[i] = new double[num_classes];
    int sum = 0;
    for (int j = 0; j < num_classes; ++j)
      sum += f_matrix[i][j];
    for (int j = 0; j < num_classes; ++j)
      matrix[i][j] = ((double) f_matrix[i][j]) / sum;
  }

  double *quality = new double[num_classes];
  for (int j = 0; j < num_classes; ++j) {
    quality[j] = matrix[0][j];
    for (int i = 1; i < num_classes; ++i) {
      if (matrix[i][j] < quality[j]) {
        quality[j] = matrix[i][j];
      }
    }
  }

  int best = 0;
  for (int i = 1; i < num_classes; ++i) {
    if (quality[i] > quality[best]) {
      best = i;
    }
  }

  lower = quality[best];
  upper = matrix[0][best];
  for (int i = 1; i < num_classes; ++i) {
    if (matrix[i][best] > upper) {
      upper = matrix[i][best];
    }
  }

  predict_label = label[best];

  delete[] quality;
  for (int i = 0; i < num_classes; ++i) {
    delete[] f_matrix[i];
    delete[] matrix[i];
  }
  delete[] f_matrix;
  delete[] matrix;

  return predict_label;
}

int SaveModel(const char *model_file_name, const struct Model *model)
{
  return 0;
}

// struct Model *LoadModel(const char *model_file_name)
// {

// }