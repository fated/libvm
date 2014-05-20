#include "utilities.h"
#include "knn.h"
#include <iostream>
#include <fstream>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *train_file_name, char *test_file_name, char *output_file_name);

struct KNNParameter knn_parameter;

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

    predict_label = KNN(train, test->x[i], knn_parameter.num_neighbors);
    output_file << predict_label << '\n';
    if (predict_label == test->y[i]) {
      ++num_correct;
    }
  }

  printf("%g%% (%d/%d) \n", 100.0*num_correct/test->l, num_correct, test->l);
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

  knn_parameter.num_neighbors = 1;

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-')
      break;
    if ((i+2) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 'k':
        ++i;
        knn_parameter.num_neighbors = atoi(argv[i]);
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