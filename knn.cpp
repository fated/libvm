#include "utilities.h"
#include "knn.h"
#include <cmath>

double KNN(struct Problem *train, struct Node *x, const int num_neighbors)
{
  double neighbors[num_neighbors];
  double labels[num_neighbors];

  for (int i = 0; i < num_neighbors; ++i) {
    neighbors[i] = -1;
    labels[i] = 0;
  }
  for (int i = 0; i < train->l; ++i) {
    double dist = CalcDist(train->x[i], x);
    int index = CompareDist(neighbors, dist, num_neighbors);
    if (index < num_neighbors) {
      InsertLabel(labels, train->y[i], num_neighbors, index);
    }
  }
  double predict_label = FindMostFrequent(labels, num_neighbors);

  return predict_label;
}

double CalcDist(struct Node *x1, struct Node *x2)
{
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

  return sqrt(sum);
}

int CompareDist(double *neighbors, double dist, int num_neighbors)
{
  int i = 0;

  while (i < num_neighbors) {
    if (dist < neighbors[i] || neighbors[i] == -1)
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

void InsertLabel(double *labels, double label, int num_neighbors, int index)
{
  for (int i = num_neighbors-1; i > index; --i)
    labels[i] = labels[i-1];
  labels[index] = label;

  return;
}