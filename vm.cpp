#include "utilities.h"
#include "knn.h"
#include "vm.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

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
  // int num_categories = param.num_categories;
  int *labels = model->labels;
  int *alter_labels = new int[l];
  int **f_matrix = new int*[num_classes];
  double predict_label;

  for (int i = 0; i < num_classes; ++i) {
    for (int j = 0; j < l; ++j) {
      if (labels[i] == train->y[j]) {
        alter_labels[j] = i;
      }
    }
  }

  for (int i = 0; i < num_classes; ++i) {
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
        ++f_matrix[i][alter_labels[j]];
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
  delete[] alter_labels;

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

void OnlinePredict(const struct Problem *prob, const struct Parameter *param, double *predict_labels, int *indices, double *lower_bounds, double *upper_bounds)
{
  int l = prob->l;
  int num_neighbors = param->knn_param.num_neighbors;
  int num_classes = 0;
  int *alter_labels = new int[l];
  std::vector<int> labels;

  for (int i = 0; i < l; ++i) {
    indices[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices, indices+l, g);

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

  int this_label = (int) prob->y[indices[0]];
  labels.push_back(this_label);
  alter_labels[0] = 0;
  num_classes = 1;

  for (int i = 1; i < l; ++i) {
    if (num_classes == 1)
      std::cerr << "WARNING: training set only has one class. See README for details." << std::endl;

    int **f_matrix = new int*[num_classes];

    for (int j = 0; j < num_classes; ++j) {
      f_matrix[j] = new int[num_classes];
      for (int k = 0; k < num_classes; ++k) {
        f_matrix[j][k] = 0;
      }

      double **dist_neighbors_ = new double*[i+1];
      int **label_neighbors_ = new int*[i+1];

      for (int j = 0; j < i; ++j) {
        clone(dist_neighbors_[j], dist_neighbors[j], num_neighbors);
        clone(label_neighbors_[j], label_neighbors[j], num_neighbors);
      }
      dist_neighbors_[i] = new double[num_neighbors];
      label_neighbors_[i] = new int[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        dist_neighbors_[i][j] = INF;
        label_neighbors_[i][j] = -1;
      }

      for (int k = 0; k < i; ++k) {
        double dist = CalcDist(prob->x[indices[k]], prob->x[indices[i]]);
        int index;

        index = CompareDist(dist_neighbors_[i], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors_[i], alter_labels[k], num_neighbors, index);
        }
        index = CompareDist(dist_neighbors_[k], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors_[k], j, num_neighbors, index);
        }
      }
      for (int k = 0; k <= i; ++k) {
        categories[k] = FindMostFrequent(label_neighbors_[k], num_neighbors);
      }

      for (int k = 0; k < i; ++k) {
        if (categories[k] == categories[i]) {
          ++f_matrix[j][alter_labels[k]];
        }
      }
      f_matrix[j][j]++;

      for (int j = 0; j < num_neighbors; ++j) {
        dist_neighbors[i][j] = dist_neighbors_[i][j];
        label_neighbors[i][j] = label_neighbors_[i][j];
      }
      for (int j = 0; j < i+1; ++j) {
        delete[] dist_neighbors_[j];
        delete[] label_neighbors_[j];
      }
      delete[] dist_neighbors_;
      delete[] label_neighbors_;
    }

    double **matrix = new double*[num_classes];
    for (int j = 0; j < num_classes; ++j) {
      matrix[j] = new double[num_classes];
      int sum = 0;
      for (int k = 0; k < num_classes; ++k)
        sum += f_matrix[j][k];
      for (int k = 0; k < num_classes; ++k)
        matrix[j][k] = ((double) f_matrix[j][k]) / sum;
    }

    double *quality = new double[num_classes];
    for (int j = 0; j < num_classes; ++j) {
      quality[j] = matrix[0][j];
      for (int k = 1; k < num_classes; ++k) {
        if (matrix[k][j] < quality[j]) {
          quality[j] = matrix[k][j];
        }
      }
    }

    int best = 0;
    for (int j = 1; j < num_classes; ++j) {
      if (quality[j] > quality[best]) {
        best = j;
      }
    }

    lower_bounds[i] = quality[best];
    upper_bounds[i] = matrix[0][best];
    for (int j = 1; j < num_classes; ++j) {
      if (matrix[j][best] > upper_bounds[i]) {
        upper_bounds[i] = matrix[j][best];
      }
    }

    predict_labels[i] = labels[(std::size_t)best];

    delete[] quality;
    for (int j = 0; j < num_classes; ++j) {
      delete[] f_matrix[j];
      delete[] matrix[j];
    }
    delete[] f_matrix;
    delete[] matrix;

    this_label = (int) prob->y[indices[i]];
    std::size_t j;
    for (j = 0; j < num_classes; ++j) {
      if (this_label == labels[j]) {
        break;
      }
    }
    alter_labels[i] = (int) j;
    if (j == num_classes) {
      labels.push_back(this_label);
      ++num_classes;
    }

    for (int j = 0; j < i; ++j) {
      double dist = CalcDist(prob->x[indices[j]], prob->x[indices[i]]);
      int index = CompareDist(dist_neighbors[j], dist, num_neighbors);
      if (index < num_neighbors) {
        InsertLabel(label_neighbors[j], alter_labels[i], num_neighbors, index);
      }
    }

  }

  for (int i = 0; i < l; ++i) {
    delete[] dist_neighbors[i];
    delete[] label_neighbors[i];
  }

  delete[] dist_neighbors;
  delete[] label_neighbors;
  delete[] categories;
  delete[] alter_labels;
  std::vector<int>(labels).swap(labels);

  return;
}

int SaveModel(const char *model_file_name, const struct Model *model)
{
  std::ofstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return -1;
  }

  const Parameter& param = model->param;
  int num_neighbors = param.knn_param.num_neighbors;

  model_file << "num_neighbors " << num_neighbors << '\n';

  int num_classes = model->num_classes;
  int l = model->l;
  model_file << "num_classes " << num_classes << '\n';
  model_file << "num_examples " << l << '\n';

  if (model->labels) {
    model_file << "labels";
    for (int i = 0; i < num_classes; ++i) {
      model_file << ' ' << model->labels[i];
    }
    model_file << '\n';
  }

  if (model->categories) {
    model_file << "categories";
    for (int i = 0; i < l; ++i) {
      model_file << ' ' << model->categories[i];
    }
    model_file << '\n';
  }

  if (model->dist_neighbors) {
    model_file << "dist_neighbors\n";
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < num_neighbors; ++j) {
        model_file << model->dist_neighbors[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  if (model->label_neighbors) {
    model_file << "label_neighbors\n";
    for (int i = 0; i < l; ++i) {
      for (int j = 0; j < num_neighbors; ++j) {
        model_file << model->label_neighbors[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  model_file.close();
  return 0;
}

struct Model *LoadModel(const char *model_file_name)
{
  std::ifstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return NULL;
  }

  Model *model = new Model;

  Parameter& param = model->param;
  model->labels = NULL;
  model->categories = NULL;
  model->dist_neighbors = NULL;
  model->label_neighbors = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (strcmp(cmd, "num_neighbors") == 0) {
      model_file >> param.knn_param.num_neighbors;
    } else
    if (strcmp(cmd, "num_classes") == 0) {
      model_file >> model->num_classes;
    } else
    if (strcmp(cmd, "num_examples") == 0) {
      model_file >> model->l;
    } else
    if (strcmp(cmd, "labels") == 0) {
      int n = model->num_classes;
      model->labels = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->labels[i];
      }
    } else
    if (strcmp(cmd, "categories") == 0) {
      int l = model->l;
      model->categories = new int[l];
      for (int i = 0; i < l; ++i) {
        model_file >> model->categories[i];
      }
    } else
    if (strcmp(cmd, "dist_neighbors") == 0) {
      int n = param.knn_param.num_neighbors;
      int l = model->l;
      model->dist_neighbors = new double*[l];
      for (int i = 0; i < l; ++i) {
        model->dist_neighbors[i] = new double[n];
        for (int j = 0; j < n; ++j) {
          model_file >> model->dist_neighbors[i][j];
        }
      }
    } else
    if (strcmp(cmd, "label_neighbors") == 0) {
      int n = param.knn_param.num_neighbors;
      int l = model->l;
      model->label_neighbors = new int*[l];
      for (int i = 0; i < l; ++i) {
        model->label_neighbors[i] = new int[n];
        for (int j = 0; j < n; ++j) {
          model_file >> model->label_neighbors[i][j];
        }
      }
      break;
    } else {
      std::cerr << "Unknown text in model file: " << cmd << std::endl;
      FreeModel(model);
      model_file.close();
      return NULL;
    }
  }
  model_file.close();

  return model;
}

void FreeModel(struct Model *model)
{
  if (model->labels != NULL) {
    delete[] model->labels;
  }

  if (model->categories != NULL) {
    delete[] model->categories;
  }

  if (model->dist_neighbors != NULL && model->label_neighbors != NULL) {
    for (int i = 0; i < model->l; ++i) {
      delete[] model->dist_neighbors[i];
      delete[] model->label_neighbors[i];
    }
    delete[] model->dist_neighbors;
    delete[] model->label_neighbors;
  }
  delete model;

  return;
}

const char *CheckParameter(const struct Parameter *param)
{
  if (param->knn_param.num_neighbors < 1) {
    return "num_neighbors should be greater than 0";
  }

  if (param->save_model == 1 && param->load_model == 1) {
    return "cannot save and load model at the same time";
  }

  return NULL;
}