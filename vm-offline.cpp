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

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *train_file_name, char *test_file_name, char *output_file_name);
double KNN(struct Problem *train, struct Node *x, const int num_neighbors);
double CalcDist(struct Node *x1, struct Node *x2);
int CompareDist(double *neighbors, double dist, int num_neighbors);
void InsertLabel(double *labels, double label, int num_neighbors, int index);

int num_neighbors = 1;

int main(int argc, char *argv[])
{
  char train_file_name[1024];
  char test_file_name[1024];
  char output_file_name[1024];
  struct Problem *train, *test;
  int num_correct = 0;

  ParseCommandLine(argc, argv, train_file_name, test_file_name, output_file_name);
  train = ReadProblem(train_file_name);
  test = ReadProblem(test_file_name);

  std::ofstream output_file(output_file_name);

  for (int i = 0; i < test->l; ++i) {
    double predict_label;

    predict_label = KNN(train, test->x[i], num_neighbors);
    output_file << predict_label << '\n';
    if (predict_label == test->y[i]) {
      ++num_correct;
    }
  }

  std::cout << num_correct << '/' << test->l << std::endl;
  output_file.close();

  return 0;
}

void ExitWithHelp()
{
  std::cout << "Usage: vm-offline [options] train_file test_file [output_file]\n"
            << "options:\n"
            << "  -k num_neighbors : set number of neighbors in kNN (default 1)"
            << std::endl;
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *train_file_name, char *test_file_name, char *output_file_name)
{
  int i;

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-')
      break;
    if ((i+2) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 'k':
        ++i;
        num_neighbors = atoi(argv[i]);
        break;
      default:
        std::cout << "Unknown option: -" << argv[i][1] << std::endl;
        ExitWithHelp();
    }
  }
  if ((i+1) >= argc)
    ExitWithHelp();
  strcpy(train_file_name, argv[i]);
  strcpy(test_file_name, argv[i+1]);
  if ((i+2) < argc) {
    strcpy(output_file_name, argv[i+2]);
  } else {
    char *p = strrchr(argv[i+1],'/');
    if (p == NULL) {
      p = argv[i+1];
    } else {
      ++p;
    }
    sprintf(output_file_name, "%s_output", p);
  }

  return;
}

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