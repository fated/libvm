#include "utilities.h"
#include "knn.h"
#include "vm.h"
#include <iostream>
#include <fstream>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *train_file_name, char *test_file_name, char *output_file_name, char *model_file_name);

struct Parameter param;

int main(int argc, char *argv[])
{
  char train_file_name[256];
  char test_file_name[256];
  char output_file_name[256];
  char model_file_name[256];
  struct Problem *train, *test;
  struct Model *model;
  int num_correct = 0;
  double avg_lower_bound = 0, avg_upper_bound = 0;

  ParseCommandLine(argc, argv, train_file_name, test_file_name, output_file_name, model_file_name);
  train = ReadProblem(train_file_name);
  test = ReadProblem(test_file_name);

  std::ofstream output_file(output_file_name);

  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  model = TrainVM(train, &param);

  if (param.save_model == 1) {
    if (SaveModel(model_file_name, model) != 0) {
      std::cerr << "Unable to save model file" << std::endl;
    }
  }

  for (int i = 0; i < test->l; ++i) {
    double predict_label, lower_bound, upper_bound;

    predict_label = PredictVM(train, model, test->x[i], lower_bound, upper_bound);
    avg_lower_bound += lower_bound;
    avg_upper_bound += upper_bound;

    output_file << predict_label << ' ' << lower_bound << ' ' << upper_bound << '\n';
    if (predict_label == test->y[i]) {
      ++num_correct;
    }
  }
  avg_lower_bound /= test->l;
  avg_upper_bound /= test->l;

  printf("%g%% (%d/%d) [%.3f%%, %.3f%%]\n", 100.0*num_correct/test->l, num_correct, test->l,
      100*avg_lower_bound, 100*avg_upper_bound);
  output_file.close();

  FreeProblem(train);
  FreeProblem(test);
  FreeModel(model);

  return 0;
}

void ExitWithHelp()
{
  std::cout << "Usage: vm-offline [options] train_file test_file [output_file]\n"
            << "options:\n"
            << "  -k num_neighbors : set number of neighbors in kNN (default 1)\n"
            << "  -s model_file_name : set model file name\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *train_file_name, char *test_file_name, char *output_file_name, char *model_file_name)
{
  int i;

  param.knn_param.num_neighbors = 1;
  param.save_model = 0;

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-')
      break;
    if ((i+2) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 'k':
        ++i;
        param.knn_param.num_neighbors = atoi(argv[i]);
        break;
      case 's':
        ++i;
        param.save_model = 1;
        strcpy(model_file_name, argv[i]);
        break;
      default:
        std::cerr << "Unknown option: -" << argv[i][1] << std::endl;
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