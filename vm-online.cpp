#include "vm.h"
#include <iostream>
#include <fstream>

void ExitWithHelp();
void ParseCommandLine(int argc, char *argv[], char *data_file_name, char *output_file_name);

struct Parameter param;

int main(int argc, char *argv[]) {
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

  if (param.taxonomy_type == SVM_EL ||
      param.taxonomy_type == SVM_ES ||
      param.taxonomy_type == SVM_KM) {
    param.svm_param->gamma = 1.0 / prob->max_index;
  }

  std::ofstream output_file(output_file_name);

  if (!output_file.is_open()) {
    std::cerr << "Unable to open output file: " << output_file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  predict_labels = new double[prob->num_ex];
  lower_bounds = new double[prob->num_ex];
  upper_bounds = new double[prob->num_ex];
  indices = new int[prob->num_ex];

  std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::high_resolution_clock::now();

  OnlinePredict(prob, &param, predict_labels, indices, lower_bounds, upper_bounds);

  std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::high_resolution_clock::now();

  output_file << prob->y[indices[0]] << '\n';

  for (int i = 1; i < prob->num_ex; ++i) {
    avg_lower_bound += lower_bounds[i];
    avg_upper_bound += upper_bounds[i];

    output_file << prob->y[indices[i]] << ' ' << predict_labels[i] << ' ' << lower_bounds[i] << ' ' << upper_bounds[i] << '\n';
    if (predict_labels[i] == prob->y[indices[i]]) {
      ++num_correct;
    }
  }
  avg_lower_bound /= prob->num_ex - 1;
  avg_upper_bound /= prob->num_ex - 1;

  printf("Accuracy: %g%% (%d/%d) Probabilities: [%.3f%%, %.3f%%]\n", 100.0*num_correct/(prob->num_ex-1), num_correct, prob->num_ex-1,
      100*avg_lower_bound, 100*avg_upper_bound);
  output_file.close();

  std::cout << "Time cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0 << " s\n";

  FreeProblem(prob);
  delete[] predict_labels;
  delete[] lower_bounds;
  delete[] upper_bounds;
  delete[] indices;

  return 0;
}

void ExitWithHelp() {
  std::cout << "Usage: vm-online [options] data_file [output_file]\n"
            << "options:\n"
            << "  -t taxonomy_type : set type of taxonomy (default 0)\n"
            << "    0 -- k-nearest neighbors\n"
            << "    1 -- support vector machine\n"
            << "  -k num_neighbors : set number of neighbors in kNN (default 1)\n"
            << "  -p : prefix of options to set parameters for SVM\n"
            << "    -ps svm_type : set type of SVM (default 0)\n"
            << "      0 -- C-SVC    (multi-class classification)\n"
            << "      1 -- nu-SVC   (multi-class classification)\n"
            << "    -pt kernel_type : set type of kernel function (default 2)\n"
            << "      0 -- linear: u'*v\n"
            << "      1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
            << "      2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
            << "      3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
            << "      4 -- precomputed kernel (kernel values in training_set_file)\n"
            << "    -pd degree : set degree in kernel function (default 3)\n"
            << "    -pg gamma : set gamma in kernel function (default 1/num_features)\n"
            << "    -pr coef0 : set coef0 in kernel function (default 0)\n"
            << "    -pc cost : set the parameter C of C-SVC (default 1)\n"
            << "    -pn nu : set the parameter nu of nu-SVC (default 0.5)\n"
            << "    -pm cachesize : set cache memory size in MB (default 100)\n"
            << "    -pe epsilon : set tolerance of termination criterion (default 0.001)\n"
            << "    -ph shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
            << "    -pwi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
            << "    -pq : quiet mode (no outputs)\n";
  exit(EXIT_FAILURE);
}

void ParseCommandLine(int argc, char **argv, char *data_file_name, char *output_file_name) {
  int i;

  param.knn_param.num_neighbors = 1;
  param.svm_param = NULL;
  param.taxonomy_type = KNN;
  param.save_model = 0;
  param.load_model = 0;

  for (i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') break;
    if ((i+1) >= argc)
      ExitWithHelp();
    switch (argv[i][1]) {
      case 't': {
        ++i;
        param.taxonomy_type = atoi(argv[i]);
        if (param.taxonomy_type == SVM_EL ||
            param.taxonomy_type == SVM_ES ||
            param.taxonomy_type == SVM_KM) {
          param.svm_param = new SVMParameter;
          InitSVMParam(param.svm_param);
        }
        break;
      }
      case 'k': {
        ++i;
        param.knn_param.num_neighbors = atoi(argv[i]);
        break;
      }
      case 'c': {
        ++i;
        param.num_categories = atoi(argv[i]);
        break;
      }
      case 'p': {
        if (argv[i][2]) {
          switch (argv[i][2]) {
            case 's': {
              ++i;
              param.svm_param->svm_type = atoi(argv[i]);
              break;
            }
            case 't': {
              ++i;
              param.svm_param->kernel_type = atoi(argv[i]);
              break;
            }
            case 'd': {
              ++i;
              param.svm_param->degree = atoi(argv[i]);
              break;
            }
            case 'g': {
              ++i;
              param.svm_param->gamma = atof(argv[i]);
              break;
            }
            case 'r': {
              ++i;
              param.svm_param->coef0 = atof(argv[i]);
              break;
            }
            case 'n': {
              ++i;
              param.svm_param->nu = atof(argv[i]);
              break;
            }
            case 'm': {
              ++i;
              param.svm_param->cache_size = atof(argv[i]);
              break;
            }
            case 'c': {
              ++i;
              param.svm_param->C = atof(argv[i]);
              break;
            }
            case 'e': {
              ++i;
              param.svm_param->eps = atof(argv[i]);
              break;
            }
            case 'h': {
              ++i;
              param.svm_param->shrinking = atoi(argv[i]);
              break;
            }
            case 'q': {
              SetPrintNull();
              break;
            }
            case 'w': {  // weights [option]: '-w1' means weight of '1'
              ++i;
              ++param.svm_param->num_weights;
              param.svm_param->weight_labels = (int *)realloc(param.svm_param->weight_labels, sizeof(int)*static_cast<unsigned long int>(param.svm_param->num_weights));
              param.svm_param->weights = (double *)realloc(param.svm_param->weights, sizeof(double)*static_cast<unsigned long int>(param.svm_param->num_weights));
              param.svm_param->weight_labels[param.svm_param->num_weights-1] = atoi(&argv[i-1][3]); // get and convert 'i' to int
              param.svm_param->weights[param.svm_param->num_weights-1] = atof(argv[i]);
              break;
              // TODO: change realloc function
            }
            default: {
              std::cerr << "Unknown SVM option: " << argv[i] << std::endl;
              ExitWithHelp();
            }
          }
        }
        break;
      }
      default: {
        std::cerr << "Unknown option: -" << argv[i][1] << std::endl;
        ExitWithHelp();
      }
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