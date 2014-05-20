#include "utilities.h"
#include <iostream>
#include <string>
#include <cmath>

double KNN(struct Problem *train, struct Node *x);
double CalcDist(struct Node *x1, struct Node *x2);

int main(int argc, char *argv[])
{
  char *train_file_name = argv[1];
  char *test_file_name = argv[2];
  struct Problem *train, *test;
  int error = 0;

  train = ReadProblem(train_file_name);
  test = ReadProblem(test_file_name);

  for (int i = 0; i < test->l; ++i) {
    double predict_label;

    predict_label = KNN(train, test->x[i]);
    if (predict_label != test->y[i]) {
      ++error;
    }
  }

  std::cout << error << '/' << test->l << std::endl;

  return 0;
}

double KNN(struct Problem *train, struct Node *x)
{
  int l = train->l, index = -1;
  double min_d = -1;

  for (int i = 0; i < l; ++i) {
    double d = CalcDist(train->x[i], x);
    if (d < min_d || min_d == -1) {
      min_d = d;
      index = i;
    }
  }

  return train->y[index];
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