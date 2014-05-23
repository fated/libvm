#include "utilities.h"
#include "knn.h"
#include "vm.h"
#include <iostream>
#include <fstream>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *data_file_name, char *output_file_name);

struct Parameter param;

int main(int argc, char *argv[])
{
  char data_file_name[256];
  char output_file_name[256];
  struct Problem *prob;
  int num_correct = 0;
  int *indices = NULL;
  double avg_lower_bound = 0, avg_upper_bound = 0;
  double *predict_labels = NULL, *lower_bounds = NULL, *upper_bounds = NULL;
  const char *error_message;

  ParseCommandLine(argc, argv, data_file_name, output_file_name);
  error_message = CheckParameter(&param);

  if (error_message != NULL) {
    std::cout << error_message << std::endl;
    exit(EXIT_FAILURE);
  }

  prob = ReadProblem(data_file_name);

  std::ofstream output_file(output_file_name);

  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  predict_labels = new double[prob->l];
  lower_bounds = new double[prob->l];
  upper_bounds = new double[prob->l];
  indices = new int[prob->l];

  OnlinePredict(prob, &param, predict_labels, indices, lower_bounds, upper_bounds);

  output_file << prob->y[indices[0]] << '\n';

  for (int i = 1; i < prob->l; ++i) {
    avg_lower_bound += lower_bounds[i];
    avg_upper_bound += upper_bounds[i];

    output_file << prob->y[indices[i]] << ' ' << predict_labels[i] << ' ' << lower_bounds[i] << ' ' << upper_bounds[i] << '\n';
    if (predict_labels[i] == prob->y[indices[i]]) {
      ++num_correct;
    }
  }
  avg_lower_bound /= prob->l - 1;
  avg_upper_bound /= prob->l - 1;

  printf("%g%% (%d/%d) [%.3f%%, %.3f%%]\n", 100.0*num_correct/(prob->l-1), num_correct, prob->l-1,
      100*avg_lower_bound, 100*avg_upper_bound);
  output_file.close();

  FreeProblem(prob);
  delete[] predict_labels;
  delete[] lower_bounds;
  delete[] upper_bounds;
  delete[] indices;

  return 0;
}

void ExitWithHelp()
{
  std::cout << "Usage: vm-online [options] data_file [output_file]\n"
            << "options:\n"
            << "  -k num_neighbors : set number of neighbors in kNN (default 1)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *data_file_name, char *output_file_name)
{
  int i;

  param.knn_param.num_neighbors = 1;
  param.save_model = 0;
  param.load_model = 0;

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-')
      break;
    if ((i+1) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 'k':
        ++i;
        param.knn_param.num_neighbors = atoi(argv[i]);
        break;
      default:
        std::cerr << "Unknown option: -" << argv[i][1] << std::endl;
        ExitWithHelp();
    }
  }
  if (i >= argc)
    ExitWithHelp();
  strcpy(data_file_name, argv[i]);
  if ((i+1) < argc) {
    strcpy(output_file_name, argv[i+1]);
  } else {
    char *p = strrchr(argv[i],'/');
    if (p == NULL) {
      p = argv[i];
    } else {
      ++p;
    }
    sprintf(output_file_name, "%s_output", p);
  }

  return;
}