#include "utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <map>

template<class T>
T FindMostFrequent(T *array, int size)
{
  std::vector<T> v(array, array+size);
  std::map<T, int> frequency_map;
  int max_frequency = 0;
  T most_frequent_element;

  for (std::size_t i = 0; i != v.size(); ++i) {
    int cur_frequency = ++frequency_map[v[i]];
    if (cur_frequency > max_frequency) {
        max_frequency = cur_frequency;
        most_frequent_element = v[i];
    }
  }

  return most_frequent_element;
}

double KNN(struct Problem *train, struct Node *x, const int num_neighbor);
double CalcDist(struct Node *x1, struct Node *x2);
int CompareDist(double *neighbors, double dist, int num_neighbor);
void InsertLabel(double *labels, double label, int num_neighbor, int index);

int main(int argc, char *argv[])
{
  std::ofstream output_file("output");
  char *train_file_name = argv[1];
  char *test_file_name = argv[2];
  struct Problem *train, *test;
  int num_correct = 0, num_neighbor = 3;

  train = ReadProblem(train_file_name);
  test = ReadProblem(test_file_name);

  for (int i = 0; i < test->l; ++i) {
    double predict_label;

    predict_label = KNN(train, test->x[i], num_neighbor);
    output_file << predict_label << '\n';
    if (predict_label == test->y[i]) {
      ++num_correct;
    }
  }

  std::cout << num_correct << '/' << test->l << std::endl;

  return 0;
}

double KNN(struct Problem *train, struct Node *x, const int num_neighbor)
{
  double neighbors[num_neighbor];
  double labels[num_neighbor];

  for (int i = 0; i < num_neighbor; ++i) {
    neighbors[i] = -1;
    labels[i] = 0;
  }
  for (int i = 0; i < train->l; ++i) {
    double dist = CalcDist(train->x[i], x);
    int index = CompareDist(neighbors, dist, num_neighbor);
    if (index < num_neighbor) {
      InsertLabel(labels, train->y[i], num_neighbor, index);
    }
  }
  double predict_label = FindMostFrequent(labels, num_neighbor);

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

int CompareDist(double *neighbors, double dist, int num_neighbor)
{
  int i = 0;

  while (i < num_neighbor) {
    if (dist < neighbors[i] || neighbors[i] == -1)
      break;
    ++i;
  }
  if (i == num_neighbor)
    return i;
  for (int j = num_neighbor-1; j > i; --j)
    neighbors[j] = neighbors[j-1];
  neighbors[i] = dist;

  return i;
}

void InsertLabel(double *labels, double label, int num_neighbor, int index)
{
  for (int i = num_neighbor-1; i > index; --i)
    labels[i] = labels[i-1];
  labels[index] = label;

  return;
}