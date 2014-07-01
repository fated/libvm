#include "knn.h"
#include <cmath>
#include <fstream>

double CalcDist(const struct Node *x1, const struct Node *x2) {
  double sum = 0;

  while (x1->index != -1 && x2->index != -1) {
    if (x1->index == x2->index) {
      sum += (x1->value - x2->value) * (x1->value - x2->value);
      ++x1;
      ++x2;
    } else {
      if(x1->index > x2->index) {
        sum += x2->value * x2->value;
        ++x2;
      } else {
        sum += x1->value * x1->value;
        ++x1;
      }
    }
  }
  if (x1->index == -1) {
    while (x2->index != -1) {
      sum += x2->value * x2->value;
      ++x2;
    }
  }
  if (x2->index == -1) {
    while (x1->index != -1) {
      sum += x1->value * x1->value;
      ++x1;
    }
  }

  return sqrt(sum);
}

int CompareDist(double *neighbors, double dist, int num_neighbors) {
  int i = 0;

  while (i < num_neighbors) {
    if (dist < neighbors[i])
      break;
    ++i;
  }
  if (i == num_neighbors)
    return i;
  for (int j = num_neighbors-1; j > i; --j)
    neighbors[j] = neighbors[j-1];
  neighbors[i] = dist;

  return i;
}

double PredictKNN(struct Problem *train, struct Node *x, const int num_neighbors) {
  double neighbors[num_neighbors];
  double labels[num_neighbors];

  for (int i = 0; i < num_neighbors; ++i) {
    neighbors[i] = INF;
    labels[i] = -1;
  }
  for (int i = 0; i < train->num_ex; ++i) {
    double dist = CalcDist(train->x[i], x);
    int index = CompareDist(neighbors, dist, num_neighbors);
    if (index < num_neighbors) {
      InsertLabel(labels, train->y[i], num_neighbors, index);
    }
  }
  double predict_label = FindMostFrequent(labels, num_neighbors);

  return predict_label;
}

int SaveKNNModel(std::ofstream &model_file, const struct KNNModel *model) {
  model_file << "num_neighbors " << model->param.num_neighbors << '\n';

  if (model->labels) {
    model_file << "labels";
    for (int i = 0; i < model->num_classes; ++i) {
      model_file << ' ' << model->labels[i];
    }
    model_file << '\n';
  }

  if (model->dist_neighbors) {
    model_file << "dist_neighbors\n";
    for (int i = 0; i < model->num_ex; ++i) {
      for (int j = 0; j < model->param.num_neighbors; ++j) {
        model_file << model->dist_neighbors[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  if (model->label_neighbors) {
    model_file << "label_neighbors\n";
    for (int i = 0; i < model->num_ex; ++i) {
      for (int j = 0; j < model->param.num_neighbors; ++j) {
        model_file << model->label_neighbors[i][j] << ' ';
      }
    }
    model_file << '\n';
  }

  return 0;
}

void InitKNNParam(struct KNNParameter *param) {
  param->num_neighbors = 1;

  return;
}