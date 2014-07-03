#include "vm.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

double CalcCombinedDecisionValues(const double *decision_values, int num_classes, int label) {
  double sum = 0;
  int k = 0, l = 0;
  for (int i = 0; i < num_classes-1; ++i) {
    for (int j = i+1; j < num_classes; ++j) {
      if (i < label && j == label) {
        sum -= decision_values[k]/2;
        ++l;
      }
      if (i == label) {
        sum += decision_values[k]/2;
        ++l;
      }
      ++k;
    }
  }

  return (sum / l) + label;
}

int GetCategory(double combined_decision_values, int num_categories) {
  int category = static_cast<int>(std::floor(combined_decision_values));
  if (category < 0) {
    category = 0;
  }
  if (category >= num_categories) {
    category = num_categories - 1;
  }

  return category;
}

Model *TrainVM(const struct Problem *train, const struct Parameter *param) {
  Model *model = new Model;
  model->param = *param;
  int num_ex = train->num_ex;

  if (param->taxonomy_type == KNN) {
    int num_neighbors = param->knn_param->num_neighbors;

    int *categories = new int[num_ex];
    for (int i = 0; i < num_ex; ++i) {
      categories[i] = -1;
    }

    model->knn_model = TrainKNN(train, param->knn_param);

    int num_categories = model->knn_model->num_classes;
    int num_classes = model->knn_model->num_classes;

    for (int i = 0; i < num_ex; ++i) {
      categories[i] = FindMostFrequent(model->knn_model->label_neighbors[i], num_neighbors);
    }

    model->num_classes = num_classes;
    model->num_ex = num_ex;
    model->num_categories = num_categories;
    model->categories = categories;
    clone(model->labels, model->knn_model->labels, num_classes);
  }

  if (param->taxonomy_type == SVM_EL ||
      param->taxonomy_type == SVM_ES ||
      param->taxonomy_type == SVM_KM) {
    int num_categories = param->num_categories;
    int *categories = new int[num_ex];
    double *combined_decision_values = new double[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      categories[i] = -1;
      combined_decision_values[i] = 0;
    }

    model->svm_model = TrainSVM(train, param->svm_param);

    int num_classes = model->svm_model->num_classes;
    if (num_classes == 1) {
      std::cerr << "WARNING: training set only has one class. See README for details." << std::endl;
    }
    if (num_categories != num_classes) {
      num_categories = num_classes;
    }

    for (int i = 0; i < num_ex; ++i) {
      double *decision_values = NULL;
      int label = 0;
      double predict_label = PredictDecisionValues(model->svm_model, train->x[i], &decision_values);
      for (int j = 0; j < num_classes; ++j) {
        if (predict_label == model->svm_model->labels[j]) {
          label = j;
          break;
        }
      }
      combined_decision_values[i] = CalcCombinedDecisionValues(decision_values, num_classes, label);
      categories[i] = GetCategory(combined_decision_values[i], num_categories);
      delete[] decision_values;
    }
    delete[] combined_decision_values;
    model->num_classes = num_classes;
    model->num_ex = num_ex;
    model->categories = categories;
    model->num_categories = num_categories;
    clone(model->labels, model->svm_model->labels, num_classes);
  }

  return model;
}

double PredictVM(const struct Problem *train, const struct Model *model, const struct Node *x, double &lower, double &upper) {
  const Parameter& param = model->param;
  int num_ex = model->num_ex;
  int num_classes = model->num_classes;
  int num_categories = model->num_categories;
  int *labels = model->labels;
  double predict_label;
  int **f_matrix = new int*[num_classes];
  int *alter_labels = new int[num_ex];

  for (int i = 0; i < num_classes; ++i) {
    for (int j = 0; j < num_ex; ++j) {
      if (labels[i] == train->y[j]) {
        alter_labels[j] = i;
      }
    }
  }

  if (param.taxonomy_type == KNN) {
    int num_neighbors = param.knn_param->num_neighbors;
    for (int i = 0; i < num_classes; ++i) {
      int *categories = new int[num_ex+1];
      double **dist_neighbors = new double*[num_ex+1];
      int **label_neighbors = new int*[num_ex+1];
      f_matrix[i] = new int[num_classes];
      for (int j = 0; j < num_classes; ++j) {
        f_matrix[i][j] = 0;
      }

      for (int j = 0; j < num_ex; ++j) {
        clone(dist_neighbors[j], model->knn_model->dist_neighbors[j], num_neighbors);
        clone(label_neighbors[j], model->knn_model->label_neighbors[j], num_neighbors);
        categories[j] = model->categories[j];
      }
      dist_neighbors[num_ex] = new double[num_neighbors];
      label_neighbors[num_ex] = new int[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        dist_neighbors[num_ex][j] = INF;
        label_neighbors[num_ex][j] = -1;
      }
      categories[num_ex] = -1;

      for (int j = 0; j < num_ex; ++j) {
        double dist = CalcDist(train->x[j], x);
        int index;
        index = CompareDist(dist_neighbors[j], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors[j], i, num_neighbors, index);
        }
        index = CompareDist(dist_neighbors[num_ex], dist, num_neighbors);
        if (index < num_neighbors) {
          InsertLabel(label_neighbors[num_ex], alter_labels[j], num_neighbors, index);
        }
      }

      for (int j = 0; j < num_ex+1; ++j) {
        categories[j] = FindMostFrequent(label_neighbors[j], num_neighbors);
      }

      for (int j = 0; j < num_ex; ++j) {
        if (categories[j] == categories[num_ex]) {
          ++f_matrix[i][alter_labels[j]];
        }
      }
      f_matrix[i][i]++;

      for (int j = 0; j < num_ex+1; ++j) {
        delete[] dist_neighbors[j];
        delete[] label_neighbors[j];
      }

      delete[] dist_neighbors;
      delete[] label_neighbors;
      delete[] categories;
    }
  }

  if (param.taxonomy_type == SVM_EL ||
      param.taxonomy_type == SVM_ES ||
      param.taxonomy_type == SVM_KM) {
    for (int i = 0; i < num_classes; ++i) {
      int *categories = new int[num_ex+1];
      f_matrix[i] = new int[num_classes];
      for (int j = 0; j < num_classes; ++j) {
        f_matrix[i][j] = 0;
      }

      for (int j = 0; j < num_ex; ++j) {
        categories[j] = model->categories[j];
      }
      categories[num_ex] = -1;

      double *decision_values = NULL;
      int label = 0;
      double predict_label = PredictDecisionValues(model->svm_model, x, &decision_values);
      for (int j = 0; j < num_classes; ++j) {
        if (predict_label == labels[j]) {
          label = j;
          break;
        }
      }
      double combined_decision_values = CalcCombinedDecisionValues(decision_values, num_classes, label);
      categories[num_ex] = GetCategory(combined_decision_values, num_categories);
      delete[] decision_values;
      for (int j = 0; j < num_ex; ++j) {
        if (categories[j] == categories[num_ex]) {
          ++f_matrix[i][alter_labels[j]];
        }
      }
      f_matrix[i][i]++;

      delete[] categories;
    }
  }

  double **matrix = new double*[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    matrix[i] = new double[num_classes];
    int sum = 0;
    for (int j = 0; j < num_classes; ++j) {
      sum += f_matrix[i][j];
    }
    for (int j = 0; j < num_classes; ++j) {
      matrix[i][j] = static_cast<double>(f_matrix[i][j]) / sum;
    }
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

  delete[] alter_labels;
  delete[] quality;
  for (int i = 0; i < num_classes; ++i) {
    delete[] f_matrix[i];
    delete[] matrix[i];
  }
  delete[] f_matrix;
  delete[] matrix;

  return predict_label;
}

void OnlinePredict(const struct Problem *prob, const struct Parameter *param,
    double *predict_labels, int *indices,
    double *lower_bounds, double *upper_bounds) {
  int num_ex = prob->num_ex;
  int num_classes = 0;

  for (int i = 0; i < num_ex; ++i) {
    indices[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices, indices+num_ex, g);

  if (param->taxonomy_type == KNN) {
    int num_neighbors = param->knn_param->num_neighbors;
    int *alter_labels = new int[num_ex];
    std::vector<int> labels;

    int *categories = new int[num_ex];
    double **dist_neighbors = new double*[num_ex];
    int **label_neighbors = new int*[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      dist_neighbors[i] = new double[num_neighbors];
      label_neighbors[i] = new int[num_neighbors];
      for (int j = 0; j < num_neighbors; ++j) {
        dist_neighbors[i][j] = INF;
        label_neighbors[i][j] = -1;
      }
      categories[i] = -1;
    }

    int this_label = static_cast<int>(prob->y[indices[0]]);
    labels.push_back(this_label);
    alter_labels[0] = 0;
    num_classes = 1;

    for (int i = 1; i < num_ex; ++i) {
      if (num_classes == 1)
        std::cerr <<
          "WARNING: training set only has one class. See README for details."
                  << std::endl;

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
          matrix[j][k] = static_cast<double>(f_matrix[j][k]) / sum;
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

      predict_labels[i] = labels[static_cast<std::size_t>(best)];

      delete[] quality;
      for (int j = 0; j < num_classes; ++j) {
        delete[] f_matrix[j];
        delete[] matrix[j];
      }
      delete[] f_matrix;
      delete[] matrix;

      this_label = static_cast<int>(prob->y[indices[i]]);
      std::size_t j;
      for (j = 0; j < num_classes; ++j) {
        if (this_label == labels[j]) break;
      }
      alter_labels[i] = static_cast<int>(j);
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

    for (int i = 0; i < num_ex; ++i) {
      delete[] dist_neighbors[i];
      delete[] label_neighbors[i];
    }

    delete[] dist_neighbors;
    delete[] label_neighbors;
    delete[] categories;
    delete[] alter_labels;
    std::vector<int>(labels).swap(labels);
  }

  if (param->taxonomy_type == SVM_EL ||
      param->taxonomy_type == SVM_ES ||
      param->taxonomy_type == SVM_KM) {
    Problem subprob;
    subprob.x = new Node*[num_ex];
    subprob.y = new double[num_ex];

    for (int i = 0; i < num_ex; ++i) {
      subprob.x[i] = prob->x[indices[i]];
      subprob.y[i] = prob->y[indices[i]];
    }

    for (int i = 1; i < num_ex; ++i) {
      subprob.num_ex = i;
      Model *submodel = TrainVM(&subprob, param);
      predict_labels[i] = PredictVM(&subprob, submodel, subprob.x[i],
                                    lower_bounds[i], upper_bounds[i]);
      FreeModel(submodel);
    }
    delete[] subprob.x;
    delete[] subprob.y;
  }

  return;
}

static const char *kTaxonomyTypeTable[] = { "knn", "svm_el", "svm_es", "svm_km", NULL };

int SaveModel(const char *model_file_name, const struct Model *model) {
  std::ofstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return -1;
  }

  const Parameter &param = model->param;

  model_file << "taxonomy_type " << kTaxonomyTypeTable[param.taxonomy_type] << '\n';
  model_file << "num_categories " << model->num_categories << '\n';

  if (param.taxonomy_type == KNN) {
    SaveKNNModel(model_file, model->knn_model);
  }
  if (param.taxonomy_type == SVM_EL ||
      param.taxonomy_type == SVM_ES ||
      param.taxonomy_type == SVM_KM) {
    SaveSVMModel(model_file, model->svm_model);
  }

  if (model->categories) {
    model_file << "categories\n";
    for (int i = 0; i < model->num_ex; ++i) {
      model_file << model->categories[i] << ' ';
    }
    model_file << '\n';
  }

  if (model_file.bad() || model_file.fail()) {
    model_file.close();
    return -1;
  }

  model_file.close();

  return 0;
}

Model *LoadModel(const char *model_file_name) {
  std::ifstream model_file(model_file_name);
  if (!model_file.is_open()) {
    std::cerr << "Unable to open model file: " << model_file_name << std::endl;
    return NULL;
  }

  Model *model = new Model;

  Parameter &param = model->param;
  param.load_model = 1;
  model->labels = NULL;
  model->categories = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "taxonomy_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kTaxonomyTypeTable[i]; ++i) {
        if (std::strcmp(kTaxonomyTypeTable[i], cmd) == 0) {
          param.taxonomy_type = i;
          break;
        }
      }
      if (kTaxonomyTypeTable[i] == NULL) {
        std::cerr << "Unknown taxonomy type.\n" << std::endl;
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "num_categories") == 0) {
      model_file >> param.num_categories;
      model->num_categories = param.num_categories;
    } else
    if (std::strcmp(cmd, "categories") == 0) {
      int num_ex = model->num_ex;
      model->categories = new int[num_ex];
      for (int i = 0; i < num_ex; ++i) {
        model_file >> model->categories[i];
      }
      break;
    } else
    if (std::strcmp(cmd, "knn_model") == 0) {
      model->knn_model = LoadKNNModel(model_file);
      if (model->knn_model == NULL) {
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
      model->num_ex = model->knn_model->num_ex;
      model->num_classes = model->knn_model->num_classes;
      clone(model->labels, model->knn_model->labels, model->num_classes);
      model->param.knn_param = &model->knn_model->param;
    } else
    if (std::strcmp(cmd, "svm_model") == 0) {
      model->svm_model = LoadSVMModel(model_file);
      if (model->svm_model == NULL) {
        FreeModel(model);
        delete model;
        model_file.close();
        return NULL;
      }
      model->num_ex = model->svm_model->num_ex;
      model->num_classes = model->svm_model->num_classes;
      clone(model->labels, model->svm_model->labels, model->num_classes);
      model->param.svm_param = &model->svm_model->param;
    } else {
      std::cerr << "Unknown text in model file: " << cmd << std::endl;
      FreeModel(model);
      delete model;
      model_file.close();
      return NULL;
    }
  }
  model_file.close();

  return model;
}

void FreeModel(struct Model *model) {
  if (model->param.taxonomy_type == KNN &&
      model->knn_model != NULL) {
    FreeKNNModel(model->knn_model);
    delete model->knn_model;
    model->knn_model = NULL;
  }

  if ((model->param.taxonomy_type == SVM_EL ||
       model->param.taxonomy_type == SVM_ES ||
       model->param.taxonomy_type == SVM_KM) &&
      model->svm_model != NULL) {
    FreeSVMModel(&(model->svm_model));
    delete model->svm_model;
    model->svm_model = NULL;
  }

  if (model->labels != NULL) {
    delete[] model->labels;
    model->labels = NULL;
  }

  if (model->categories != NULL) {
    delete[] model->categories;
    model->labels = NULL;
  }

  delete model;
  model = NULL;

  return;
}

void FreeParam(struct Parameter *param) {
  if (param->taxonomy_type == KNN &&
      param->knn_param != NULL) {
    FreeKNNParam(param->knn_param);
    param->knn_param = NULL;
  }

  if ((param->taxonomy_type == SVM_EL ||
       param->taxonomy_type == SVM_ES ||
       param->taxonomy_type == SVM_KM) &&
      param->svm_param != NULL) {
    FreeSVMParam(param->svm_param);
    param->svm_param = NULL;
  }

  return;
}

const char *CheckParameter(const struct Parameter *param) {
  if (param->save_model == 1 && param->load_model == 1) {
    return "cannot save and load model at the same time";
  }

  if (param->taxonomy_type == KNN) {
    if (param->knn_param == NULL) {
      return "no knn parameter";
    }
    return CheckKNNParameter(param->knn_param);
  }

  if (param->taxonomy_type == SVM_EL ||
      param->taxonomy_type == SVM_ES ||
      param->taxonomy_type == SVM_KM) {
    if (param->svm_param == NULL) {
      return "no svm parameter";
    }
    return CheckSVMParameter(param->svm_param);
  }

  return NULL;
}