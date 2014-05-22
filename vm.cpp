#include "utilities.h"
#include "knn.h"
#include "vm.h"
#include <iostream>
#include <cmath>

template <typename T, typename S> static inline void clone(T *&dest, S *src, int size)
{
  dest = new T[size];
  if (sizeof(T) < sizeof(S))
    std::cerr << "WARNING: destination type is smaller than source type, data will be truncated." << std::endl;
  std::copy(src, src+size, dest);
}

struct Model *TrainVM(const struct Problem *train, const struct Parameter *param)
{
  Model *model = new Model;
  model->param = *param;

  int l = train->l;
  int num_classes = 0;
  int num_neighbors = param->knn_param.num_neighbors;
  int *labels = NULL;
  int *alter_labels = new int[l];

  std::vector<int> unique_labels;
  for (int i = 0; i < l; ++i) {
    int this_label = (int) train->y[i];
    std::size_t j;
    for (j = 0; j < num_classes; ++j) {
      if (this_label == unique_labels[j]) {
        break;
      }
    }
    alter_labels[i] = (int) j;
    if (j == num_classes) {
      unique_labels.push_back(this_label);
      ++num_classes;
    }
  }
  labels = new int[num_classes];
  for (std::size_t i = 0; i < unique_labels.size(); ++i) {
    labels[i] = unique_labels[i];
  }
  std::vector<int>(unique_labels).swap(unique_labels);

  if (num_classes == 1)
    std::cerr << "WARNING: training set only has one class. See README for details." << std::endl;

  int *categories = new int[l];
  double **dist_neighbors = new double*[l];
  int **label_neighbors = new int*[l];

  for (int i = 0; i < l; ++i) {
    dist_neighbors[i] = new double[num_neighbors];
    label_neighbors[i] = new int[num_neighbors];
    for (int j = 0; j < num_neighbors; ++j) {
      dist_neighbors[i][j] = INF;
      label_neighbors[i][j] = -1;
    }
    categories[i] = -1;
  }

  for (int i = 0; i < l-1; ++i) {
    for (int j = i+1; j < l; ++j) {
      double dist = CalcDist(train->x[i], train->x[j]);
      int index;

      index = CompareDist(dist_neighbors[i], dist, num_neighbors);
      if (index < num_neighbors) {
        InsertLabel(label_neighbors[i], alter_labels[j], num_neighbors, index);
      }
      index = CompareDist(dist_neighbors[j], dist, num_neighbors);
      if (index < num_neighbors) {
        InsertLabel(label_neighbors[j], alter_labels[i], num_neighbors, index);
      }
    }
  }

  for (int i = 0; i < l; ++i) {
    categories[i] = FindMostFrequent(label_neighbors[i], num_neighbors);
  }
  delete[] alter_labels;

  model->num_classes = num_classes;
  model->l = l;
  model->labels = labels;
  model->categories = categories;
  model->dist_neighbors = dist_neighbors;
  model->label_neighbors = label_neighbors;

  return model;
}

double PredictVM(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper)
{
  const Parameter& param = model->param;
  int l = model->l;
  int num_classes = model->num_classes;
  int num_neighbors = param.knn_param.num_neighbors;
  // int num_categories = num_classes;
  int *labels = model->labels;
  int *alter_labels = new int[l];
  int **f_matrix = new int*[num_classes];
  double y, predict_label;

  for (int i = 0; i < num_classes; ++i) {
    for (int j = 0; j < l; ++j) {
      if (labels[i] == train->y[j]) {
        alter_labels[j] = i;
      }
    }
  }

  for (int i = 0; i < num_classes; ++i) {
    y = labels[i];

    int *categories = new int[l+1];
    double **dist_neighbors = new double*[l+1];
    int **label_neighbors = new int*[l+1];
    f_matrix[i] = new int[num_classes];
    for (int j = 0; j < num_classes; ++j) {
      f_matrix[i][j] = 0;
    }

    for (int j = 0; j < l; ++j) {
      clone(dist_neighbors[j], model->dist_neighbors[j], num_neighbors);
      clone(label_neighbors[j], model->label_neighbors[j], num_neighbors);
      categories[j] = model->categories[j];
    }
    dist_neighbors[l] = new double[num_neighbors];
    label_neighbors[l] = new int[num_neighbors];
    for (int j = 0; j < num_neighbors; ++j) {
      dist_neighbors[l][j] = INF;
      label_neighbors[l][j] = -1;
    }
    categories[l] = -1;

    for (int j = 0; j < l; ++j) {
      double dist = CalcDist(train->x[j], x);
      int index;

      index = CompareDist(dist_neighbors[j], dist, num_neighbors);
      if (index < num_neighbors) {
        InsertLabel(label_neighbors[j], i, num_neighbors, index);
      }
      index = CompareDist(dist_neighbors[l], dist, num_neighbors);
      if (index < num_neighbors) {
        InsertLabel(label_neighbors[l], alter_labels[j], num_neighbors, index);
      }
    }

    for (int j = 0; j < l+1; ++j) {
      categories[j] = FindMostFrequent(label_neighbors[j], num_neighbors);
    }

    for (int j = 0; j < l; ++j) {
      if (categories[j] == categories[l]) {
        f_matrix[i][alter_labels[j]]++;
        break;
      }
    }
    f_matrix[i][i]++;

    for (int j = 0; j < l+1; ++j) {
      delete[] dist_neighbors[j];
      delete[] label_neighbors[j];
    }

    delete[] dist_neighbors;
    delete[] label_neighbors;
    delete[] categories;
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

  predict_label = labels[best];

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

void FreeModel(struct Model *model)
{
  delete[] model->labels;
  delete[] model->categories;

  for (int i = 0; i < model->l; ++i) {
    delete[] model->dist_neighbors[i];
    delete[] model->label_neighbors[i];
  }
  delete[] model->dist_neighbors;
  delete[] model->label_neighbors;

  return;
}