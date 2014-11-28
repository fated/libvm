#include "mcsvm.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cstdarg>
#include <random>

int CompareNodes(const void *n1, const void *n2) {
  if (((struct Node *)n1)->value > ((struct Node *)n2)->value)
    return (-1);
  else if (((struct Node *)n1)->value < ((struct Node *)n2)->value)
    return (1);
  else
    return (0);
}

int CompareInt(const void *n1, const void *n2) {
  if (*(int*)n1 < *(int*)n2)
    return (-1);
  else if (*(int*)n1 > *(int*)n2)
    return (1);
  else
    return (0);
}

// ReducedOptimization start

static const char *kRedOptTypeNameTable[] = {
  "exact",
  "approximate",
  "binary",
  NULL
};

class RedOpt {
 public:
  RedOpt(int num_classes, const MCSVMParameter &param);
  virtual ~RedOpt();

  void set_a(double a) {
    a_ = a;
  }
  double get_a() {
    return a_;
  }
  void set_y(int y) {
    y_ = y;
  }
  int get_y() {
    return y_;
  }
  void set_b(double b, int i) {
    b_[i] = b;
  }
  double get_b(int i) {
    return b_[i];
  }
  void set_alpha(double *alpha) {
    alpha_ = alpha;
  }
  int RedOptFunction() {
    return (this->*redopt_function)();
  }
  void RedOptText() {
    Info("Reduced optimization algorithm : %s\n", kRedOptTypeNameTable[redopt_type_]);

    return;
  }

 private:
  int num_classes_;
  int y_;
  double a_;
  double *b_;
  double *alpha_;
  Node *vector_d_;

  // MCSVMParameter
  const int redopt_type_;
  const double delta_;

  void Two(double v0, double v1, int i0, int i1) {
    double temp = 0.5 * (v0-v1);
    temp = (temp < 1) ? temp : 1;
    alpha_[i0] = -temp;
    alpha_[i1] = temp;
  }
  int (RedOpt::*redopt_function)();
  int RedOptExact();
  int RedOptApprox();
  int RedOptAnalyticBinary();
  int GetMarginError(const double beta);
};

RedOpt::RedOpt(int num_classes, const MCSVMParameter &param)
    :num_classes_(num_classes),
     redopt_type_(param.redopt_type),
     delta_(param.delta) {
  switch (redopt_type_) {
    case EXACT: {
      redopt_function = &RedOpt::RedOptExact;
      break;
    }
    case APPROX: {
      redopt_function = &RedOpt::RedOptApprox;
      Info("Delta %e\n", delta_);
      break;
    }
    case BINARY: {
      redopt_function = &RedOpt::RedOptAnalyticBinary;
      break;
    }
    default: {
      // assert(false);
      break;
    }
  }
  b_ = new double[num_classes_];
  vector_d_ = new Node[num_classes_];

  RedOptText();
}

RedOpt::~RedOpt() {
  delete[] b_;
  delete[] vector_d_;
}

// solve reduced exactly, use sort
int RedOpt::RedOptExact() {
  double phi0 = 0;  // potenial functions phi(t)
  double phi1;  // potenial functions phi(t+1)
  double sum_d = 0;
  double theta;  // threshold
  int mistake_k = 0;  // no. of labels with score greater than the correct label
  int r;
  int r1;

  // pick only problematic labels
  for (int i = 0; i < num_classes_; ++i) {
    if (b_[i] > b_[y_]) {
      vector_d_[mistake_k].index = i;
      vector_d_[mistake_k].value = b_[i] / a_;
      sum_d += vector_d_[mistake_k].value;
      ++mistake_k;
    } else {  // for other labels, alpha=0
      alpha_[i] = 0;
    }
  }

  /* if no mistake labels return */
  if (mistake_k == 0) {
    return (0);
  }
  /* add correct label to list (up to constant one) */
  vector_d_[mistake_k].index = y_;
  vector_d_[mistake_k].value = b_[y_] / a_;

  /* if there are only two bad labels, solve for it */
  if (mistake_k == 1) {
    Two(vector_d_[0].value, vector_d_[1].value, vector_d_[0].index, vector_d_[1].index);
    return (2);
  }

  /* finish calculations */
  sum_d += vector_d_[mistake_k].value;
  ++vector_d_[mistake_k].value;
  ++mistake_k;

  /* sort vector_d by values */
  qsort(vector_d_, static_cast<size_t>(mistake_k), sizeof(struct Node), CompareNodes);

  /* go down the potential until sign reversal */
  for (r = 1, phi1 = 1; phi1 > 0 && r < mistake_k; ++r) {
    phi0 = phi1;
    phi1 = phi0 - r * (vector_d_[r-1].value - vector_d_[r].value);
  }

  /* theta < min vector_d.value */
  /* nu[r] = theta */
  if (phi1 > 0) {
    sum_d /= mistake_k;
    for (r = 0; r < mistake_k; ++r) {
      alpha_[vector_d_[r].index] = sum_d - vector_d_[r].value;
    }
    ++alpha_[y_];
  } else {  /* theta > min vector_d.value */
    theta = - phi0 / (--r);
    theta += vector_d_[--r].value;
    /* update tau[r] with nu[r]=theta */
    for (r1 = 0; r1 <= r; ++r1) {
      alpha_[vector_d_[r1].index] = theta - vector_d_[r1].value;
    }
    /* update tau[r]=0, nu[r]=vector[d].r */
    for ( ; r1 < mistake_k; ++r1) {
      alpha_[vector_d_[r1].index] = 0;
    }
    ++alpha_[y_];
  }

  return (mistake_k);
}

int RedOpt::RedOptApprox() {
  double old_theta = DBL_MAX;  /* threshold */
  double theta = DBL_MAX;      /* threshold */
  int mistake_k =0; /* no. of labels with score greater than the correct label */

  /* pick only problematic labels */
  for (int i = 0; i < num_classes_; ++i) {
    if (b_[i] > b_[y_]) {
      vector_d_[mistake_k].index = i;
      vector_d_[mistake_k].value = b_[i] / a_;
      ++mistake_k;
    }
    /* for other labels, alpha=0 */
    else {
      alpha_[i] = 0;
    }
  }

  /* if no mistake labels return */
  if (mistake_k == 0) {
    return (0);
  }

  /* add correct label to list (up to constant one) */
  vector_d_[mistake_k].index = y_;
  vector_d_[mistake_k].value = b_[y_] / a_;

  /* if there are only two bad labels, solve for it */
  if (mistake_k == 1) {
    Two(vector_d_[0].value, vector_d_[1].value, vector_d_[0].index, vector_d_[1].index);
    return (2);
  }

  /* finish calculations */
  ++vector_d_[mistake_k].value;
  ++mistake_k;

  /* initialize theta to be min D_r */
  for (int i = 0; i < mistake_k; ++i) {
    if (vector_d_[i].value < theta) {
      theta = vector_d_[i].value;
    }
  }

  /* loop until convergence of theta */
  while (1) {
    old_theta = theta;

    /* calculate new value of theta */
    theta = -1;
    for (int i = 0; i < mistake_k; ++i) {
      if (old_theta > vector_d_[i].value) {
        theta += old_theta;
      } else {
        theta += vector_d_[i].value;
      }
    }
    theta /= mistake_k;

    if (std::fabs((old_theta-theta)/theta) < delta_) {
      break;
    }
  }

  /* update alpha using threshold */
  for (int i = 0; i < mistake_k; ++i) {
    double temp = theta - vector_d_[i].value;
    if (temp < 0) {
      alpha_[vector_d_[i].index] = temp;
    } else {
      alpha_[vector_d_[i].index] = 0;
    }
  }
  ++alpha_[y_];

  return(mistake_k);
}

/* solve for num_classes_=2 */
int RedOpt::RedOptAnalyticBinary() {
  int y0 = 1 - y_; /* other label */
  int y1 = y_;    /* currect label */

  if (b_[y0] > b_[y1]) {
    vector_d_[y0].value = b_[y0] / a_;
    vector_d_[y1].value = b_[y1] / a_;

    Two(vector_d_[y0].value, vector_d_[y1].value, y0, y1);
    return (2);
  } else {
    alpha_[0] = 0;
    alpha_[1] = 0;
    return (0);
  }
}

int RedOpt::GetMarginError(const double beta) {
  int errors = 0;
  int i;

  for (i = 0; i < y_; ++i) {
    if (b_[i] >= b_[y_]-beta) {
      ++errors;
    }
  }

  for (++i; i < num_classes_; ++i) {
    if (b_[i] >= b_[y_]-beta) {
      ++errors;
    }
  }

  return errors;
}

// RedOpt end

// Q class begin

class SPOC_Q : public Kernel {
 public:
  SPOC_Q(const Problem &prob, const MCSVMParameter &param) : Kernel(prob.num_ex, prob.x, param.kernel_param) {
    cache_ = new Cache(prob.num_ex, static_cast<long>(param.cache_size*(1<<20)));
    QD_ = new double[prob.num_ex];
    for (int i = 0; i < prob.num_ex; ++i)
      QD_[i] = (this->*kernel_function)(i, i);
  }

  Qfloat *get_Q(int i, int len) const {
    Qfloat *data;
    int start = cache_->get_data(i, &data, len);
    if (start < len) {
      for (int j = start; j < len; ++j)
        data[j] = static_cast<Qfloat>((this->*kernel_function)(i, j));
    }
    return data;
  }

  double *get_QD() const {
    return QD_;
  }

  void SwapIndex(int i, int j) const {
    cache_->SwapIndex(i, j);
    Kernel::SwapIndex(i, j);
    std::swap(QD_[i], QD_[j]);
  }

  ~SPOC_Q() {
    delete cache_;
    delete[] QD_;
  }

 private:
  Cache *cache_;
  double *QD_;  // Q matrix Diagonal
};

// Q class ends

// Spoc start

class Spoc {
 public:
  Spoc(const Problem *prob, const MCSVMParameter *param, int *y, int num_classes);
  virtual ~Spoc();

  struct SolutionInfo {
    int total_sv;
    int *num_svs;
    int *sv_indices;
    double **tau;
  };

  Spoc::SolutionInfo *Solve();

 protected:

 private:
  const double epsilon_;
  const double epsilon0_;
  int iteration_;
  int num_ex_;
  int num_classes_;
  int num_support_pattern_;
  int num_zero_pattern_;
  int next_p_;
  int next_p_list_;
  int *y_;
  int *support_pattern_list_;
  int *zero_pattern_list_;
  int **matrix_eye_;
  double max_psi_;
  double beta_;
  double *row_matrix_f_next_p_;
  double *vector_a_;
  double *vector_b_;
  double *delta_tau_;
  double *old_tau_;
  double **matrix_f_;
  double **tau_;
  SPOC_Q *spoc_Q_;
  RedOpt *red_opt_;

  void CalcEpsilon(double epsilon);
  void ChooseNextPattern(int *pattern_list, int num_patterns);
  void UpdateMatrix(double *kernel_next_p);
  double CalcTrainError(double beta);
  int CountNumSVs();

  void PrintEpsilon(double epsilon) {
    Info("%11.5e   %7ld   %10.3e   %7.2f%%      %7.2f%%\n",
      epsilon, num_support_pattern_, max_psi_/beta_, CalcTrainError(beta_), CalcTrainError(0));

    return;
  }

  double NextEpsilon(double epsilon_cur, double epsilon) {
    double e = epsilon_cur / std::log10(iteration_);
    iteration_ += 2;

    return (std::max(e, epsilon));
  }

};

Spoc::Spoc(const Problem *prob, const MCSVMParameter *param, int *y, int num_classes)
    :epsilon_(param->epsilon),
     epsilon0_(param->epsilon0),
     iteration_(12),
     num_ex_(prob->num_ex),
     num_classes_(num_classes),
     y_(y),
     beta_(param->beta) {

  Info("\nOptimizer (SPOC) ... start\n");
  Info("Initializing ... start\n");

  Info("Requested margin (beta) %e\n", beta_);

  // allocate memory start
  // tau
  tau_ = new double*[num_ex_];
  *tau_ = new double[num_ex_ * num_classes_];
  for (int i = 1; i < num_ex_; ++i) {
    tau_[i] = tau_[i-1] + num_classes_;
  }

  // matrix_f
  matrix_f_ = new double*[num_ex_];
  *matrix_f_ = new double[num_ex_ * num_classes_];
  for (int i = 1; i < num_ex_; ++i) {
    matrix_f_[i] = matrix_f_[i-1] + num_classes_;
  }

  // matrix_eye
  matrix_eye_ = new int*[num_classes_];
  *matrix_eye_ = new int[num_classes_ * num_classes_];
  for (int i = 1; i < num_classes_; ++i) {
    matrix_eye_[i] = matrix_eye_[i-1] + num_classes_;
  }

  // delta_tau
  delta_tau_ = new double[num_classes_];

  // old_tau
  old_tau_ = new double[num_classes_];

  // vector_b
  vector_b_ = new double[num_classes_];

  // supp_pattern_list
  support_pattern_list_ = new int[num_ex_];

  // zero_pattern_list
  zero_pattern_list_ = new int[num_ex_];

  // allocate memory end

  red_opt_ = new RedOpt(num_classes_, *param);
  spoc_Q_ = new SPOC_Q(*prob, *param);

  num_support_pattern_ = 0;
  num_zero_pattern_ = 0;

  // initialize begins

  // vector_a
  vector_a_ = spoc_Q_->get_QD();

  // matrix_eye
  for (int i = 0; i < num_classes_; ++i) {
    for (int j = 0; j < num_classes_; ++j) {
      if (i != j) {
        matrix_eye_[i][j] = 0;
      } else {
        matrix_eye_[i][j] = 1;
      }
    }
  }

  // matrix_f
  for (int i = 0; i < num_ex_; ++i) {
    for (int j = 0; j < num_classes_; ++j) {
      if (y_[i] != j) {
        matrix_f_[i][j] = 0;
      } else {
        matrix_f_[i][j] = -beta_;
      }
    }
  }

  // tau
  for (int i = 0; i < num_ex_; ++i) {
    for (int j = 0 ; j < num_classes_; ++j) {
      tau_[i][j] = 0;
    }
  }

  support_pattern_list_[0] = 0;
  num_support_pattern_ = 1;

  for (int i = 1; i < num_ex_; ++i) {
    zero_pattern_list_[i-1] = i;
  }
  num_zero_pattern_ = num_ex_ - 1;
  ChooseNextPattern(support_pattern_list_, num_support_pattern_);

  Info("Initializing ... done\n");
}

Spoc::~Spoc() {
  if (matrix_f_ != NULL) {
    if (*matrix_f_ != NULL) {
      delete[] *matrix_f_;
    }
    delete[] matrix_f_;
  }
  if (matrix_eye_ != NULL) {
    if (matrix_eye_ != NULL) {
      delete[] *matrix_eye_;
    }
    delete[] matrix_eye_;
  }
  if (delta_tau_ != NULL) {
    delete[] delta_tau_;
  }
  if (old_tau_ != NULL) {
    delete[] old_tau_;
  }
  if (vector_b_ != NULL) {
    delete[] vector_b_;
  }
  if (zero_pattern_list_ != NULL) {
    delete[] zero_pattern_list_;
  }
  if (support_pattern_list_ != NULL) {
    delete[] support_pattern_list_;
  }
  if (tau_ != NULL) {
    if (*tau_ != NULL) {
      delete[] *tau_;
    }
    delete[] tau_;
  }
  delete red_opt_;
  delete spoc_Q_;
}

Spoc::SolutionInfo *Spoc::Solve() {
  double epsilon_current = epsilon0_;

  Info("Epsilon decreasing from %e to %e\n", epsilon0_, epsilon_);
  Info("\nNew Epsilon   No. SPS      Max Psi   Train Error   Margin Error\n");
  Info("-----------   -------      -------   -----------   ------------\n");

  while (max_psi_ > epsilon_ * beta_) {
    PrintEpsilon(epsilon_current);
    CalcEpsilon(epsilon_current);
    epsilon_current = NextEpsilon(epsilon_current, epsilon_);
  }
  PrintEpsilon(epsilon_);

  Info("\nNo. support pattern %d ( %d at bound )\n", num_support_pattern_, CountNumSVs());

  qsort(support_pattern_list_, static_cast<size_t>(num_support_pattern_), sizeof(int), &CompareInt);

  SolutionInfo *si = new SolutionInfo;

  si->total_sv = num_support_pattern_;
  si->num_svs = new int[num_classes_];
  si->sv_indices = new int[si->total_sv];
  si->tau = new double*[num_classes_];

  for (int i = 0; i < num_classes_; ++i) {
    si->num_svs[i] = 0;
    si->tau[i] = new double[si->total_sv];
  }

  for (int i = 0; i < si->total_sv; ++i) {
    si->sv_indices[i] = support_pattern_list_[i] + 1;
  }

  for (int i = 0; i < num_classes_; ++i) {
    for (int j = 0; j < si->total_sv; ++j) {
      si->tau[i][j] = tau_[support_pattern_list_[j]][i];
      if (tau_[support_pattern_list_[j]][i] != 0) {
        ++si->num_svs[i];
      }
    }
  }

  Info(" class\tsupport patterns per class\n");
  Info(" -----\t--------------------------\n");
  for (int i = 0; i < num_classes_; ++i) {
    Info("   %d\t    %d\n", i, si->num_svs[i]);
  }
  Info("\n");

  return si;
}

void Spoc::CalcEpsilon(double epsilon) {
  int supp_only =1;
  int cont = 1;
  int mistake_k;
  double *kernel_next_p;

  while (cont) {
    max_psi_ = 0;
    if (supp_only) {
      ChooseNextPattern(support_pattern_list_, num_support_pattern_);
    } else {
      ChooseNextPattern(zero_pattern_list_, num_zero_pattern_);
    }

    if (max_psi_ > epsilon * beta_) {
      red_opt_->set_a(vector_a_[next_p_]);
      for (int i = 0; i < num_classes_; ++i) {
        double b = matrix_f_[next_p_][i] - red_opt_->get_a() * tau_[next_p_][i];
        red_opt_->set_b(b, i);
      }
      red_opt_->set_y(y_[next_p_]);
      for (int i = 0; i < num_classes_; ++i) {
        old_tau_[i] = tau_[next_p_][i];
      }
      red_opt_->set_alpha(tau_[next_p_]);

      mistake_k = red_opt_->RedOptFunction();

      for (int i = 0; i < num_classes_; ++i) {
        delta_tau_[i] = tau_[next_p_][i] - old_tau_[i];
      }

      kernel_next_p = spoc_Q_->get_Q(next_p_, num_ex_);

      UpdateMatrix(kernel_next_p);

      if (supp_only) {
        int i;
        for (i = 0; i < num_classes_; ++i) {
          if (tau_[next_p_][i] != 0) {
            break;
          }
        }
        if (i == num_classes_) {
          zero_pattern_list_[num_zero_pattern_++] = next_p_;
          support_pattern_list_[next_p_list_] = support_pattern_list_[--num_support_pattern_];
        }
      } else {
        support_pattern_list_[num_support_pattern_++] = next_p_;
        zero_pattern_list_[next_p_list_] = zero_pattern_list_[--num_zero_pattern_];
        supp_only = 1;
      }
    } else {
      if (supp_only)
        supp_only = 0;
      else
        cont = 0;
    }
  }

  return;
}

void Spoc::ChooseNextPattern(int *pattern_list, int num_patterns) {
  // psi : KKT value of example
  // psi1 : max_r matrix_f[i][j]
  // psi0 : min_{j, tau[i][j]<delta[yi][j]}  matrix_f[i][j]
  int p = 0;
  double *matrix_f_ptr;

  for (int i = 0; i < num_patterns; ++i) {
    double psi1 = -DBL_MAX;
    double psi0 = DBL_MAX;

    p = pattern_list[i];
    matrix_f_ptr = matrix_f_[p];

    for (int j = 0; j < num_classes_; ++j) {
      if (*matrix_f_ptr > psi1) {
        psi1 = *matrix_f_ptr;
      }

      if (*matrix_f_ptr < psi0) {
        if (tau_[p][j] < matrix_eye_[y_[p]][j]) {
          psi0 = *matrix_f_ptr;
        }
      }

      ++matrix_f_ptr;
    }

    double psi = psi1 - psi0;

    if (psi > max_psi_) {
      next_p_list_ = i;
      max_psi_ = psi;
    }
  }
  next_p_ = pattern_list[next_p_list_];
  row_matrix_f_next_p_ = matrix_f_[p];

  return;
}

void Spoc::UpdateMatrix(double *kernel_next_p) {
  double *delta_tau_ptr = delta_tau_;
  double *kernel_next_p_ptr;

  for (int j = 0; j < num_classes_; ++j) {
    if (*delta_tau_ptr != 0) {
      kernel_next_p_ptr = kernel_next_p;
      for (int i = 0; i < num_ex_; ++i) {
        matrix_f_[i][j] += (*delta_tau_ptr) * (*kernel_next_p_ptr);
        ++kernel_next_p_ptr;
      }
    }
    ++delta_tau_ptr;
  }

  return;
}

double Spoc::CalcTrainError(double beta) {
  int errors = 0;

  for (int i = 0; i < num_ex_; ++i) {
    int j;
    double max = -DBL_MAX;
    for (j = 0; j < y_[i]; ++j) {
      if (matrix_f_[i][j] > max) {
        max = matrix_f_[i][j];
      }
    }
    for (++j; j < num_classes_; ++j) {
      if (matrix_f_[i][j] > max) {
        max = matrix_f_[i][j];
      }
    }
    if ((max-beta) >= matrix_f_[i][y_[i]]) {
      ++errors;
    }
  }

  return (100.0*errors/(static_cast<double>(num_ex_)));
}

int Spoc::CountNumSVs() {
  int n = 0;

  for (int i = 0; i < num_ex_; ++i)
    if (tau_[i][y_[i]] == 1) {
      ++n;
    }

  return n;
}

// Spoc class end

MCSVMModel *TrainMCSVM(const struct Problem *prob, const struct MCSVMParameter *param) {
  MCSVMModel *model = new MCSVMModel;
  model->param = *param;

  // calc labels
  int num_ex = prob->num_ex;
  int num_classes = 0;
  int *labels = NULL;
  int *alter_labels = new int[num_ex];
  std::vector<int> unique_labels;

  for (int i = 0; i < num_ex; ++i) {
    int this_label = static_cast<int>(prob->y[i]);
    std::size_t j;
    for (j = 0; j < num_classes; ++j) {
      if (this_label == unique_labels[j]) {
        break;
      }
    }
    alter_labels[i] = static_cast<int>(j);
    if (j == num_classes) {
      unique_labels.push_back(this_label);
      ++num_classes;
    }
  }
  labels = new int[num_classes];
  for (std::size_t i = 0; i < unique_labels.size(); ++i) {
    labels[i] = unique_labels[i];
  }
  std::vector<int>(unique_labels).swap(unique_labels);

  if (num_classes == 1) {
    Info("WARNING: training data in only one class. See README for details.\n");
  }

  // train MSCVM model
  Info("\nCrammer and Singer's Multi-Class SVM Train\n");
  Info("%d examples,  %d classes\n", num_ex, num_classes);
  Spoc s(prob, param, alter_labels, num_classes);
  Spoc::SolutionInfo *si = s.Solve();

  // build output
  model->total_sv = si->total_sv;
  model->sv_indices = si->sv_indices;
  model->num_svs = si->num_svs;
  model->tau = si->tau;
  model->svs = new Node*[model->total_sv];

  for (int i = 0; i < model->total_sv; ++i) {
    model->svs[i] = prob->x[model->sv_indices[i]-1];
  }
  model->num_ex = num_ex;
  model->num_classes = num_classes;
  model->labels = labels;

  delete[] alter_labels;

  return (model);
}

double *PredictMCSVMValues(const struct MCSVMModel *model, const struct Node *x) {
  int num_classes = model->num_classes;
  int total_sv = model->total_sv;

  double *sim_scores = new double[num_classes];
  double *kernel_values = new double[total_sv];

  for (int i = 0; i < total_sv; ++i) {
    kernel_values[i] = Kernel::KernelFunction(x, model->svs[i], model->param.kernel_param);
  }

  for (int i = 0; i < num_classes; ++i) {
    sim_scores[i] = 0;
    for (int j = 0; j < total_sv; ++j) {
      if (model->tau[i][j] != 0) {
        sim_scores[i] += model->tau[i][j] * kernel_values[j];
      }
    }
  }

  delete[] kernel_values;

  return sim_scores;
}

int PredictMCSVM(const struct MCSVMModel *model, const struct Node *x, int *num_max_sim_score_ret) {
  int num_classes = model->num_classes;
  double max_sim_score = -DBL_MAX;
  int predicted_label = -1;
  int num_max_sim_score = 0;

  double *sim_scores = PredictMCSVMValues(model, x);

  for (int i = 0; i < num_classes; ++i) {
    if (sim_scores[i] > max_sim_score) {
      max_sim_score = sim_scores[i];
      num_max_sim_score = 1;
      predicted_label = i;
    } else {
      if (sim_scores[i] == max_sim_score) {
        ++num_max_sim_score;
      }
    }
  }

  delete[] sim_scores;
  *num_max_sim_score_ret = num_max_sim_score;

  return model->labels[predicted_label];
}

static const char *kRedOptTypeTable[] = { "exact", "approx", "binary", NULL };

static const char *kKernelTypeTable[] = { "linear", "polynomial", "rbf", "sigmoid", "precomputed", NULL };

int SaveMCSVMModel(std::ofstream &model_file, const struct MCSVMModel *model) {
  const MCSVMParameter &param = model->param;

  model_file << "mcsvm_model\n";
  model_file << "redopt_type " << kRedOptTypeTable[param.redopt_type] << '\n';
  model_file << "kernel_type " << kKernelTypeTable[param.kernel_param->kernel_type] << '\n';

  if (param.kernel_param->kernel_type == POLY) {
    model_file << "degree " << param.kernel_param->degree << '\n';
  }
  if (param.kernel_param->kernel_type == POLY ||
      param.kernel_param->kernel_type == RBF  ||
      param.kernel_param->kernel_type == SIGMOID) {
    model_file << "gamma " << param.kernel_param->gamma << '\n';
  }
  if (param.kernel_param->kernel_type == POLY ||
      param.kernel_param->kernel_type == SIGMOID) {
    model_file << "coef0 " << param.kernel_param->coef0 << '\n';
  }

  int num_classes = model->num_classes;
  int total_sv = model->total_sv;
  model_file << "num_examples " << model->num_ex << '\n';
  model_file << "num_classes " << num_classes << '\n';
  model_file << "total_SV " << total_sv << '\n';

  if (model->labels) {
    model_file << "labels";
    for (int i = 0; i < num_classes; ++i)
      model_file << ' ' << model->labels[i];
    model_file << '\n';
  }

  if (model->num_svs) {
    model_file << "num_SVs";
    for (int i = 0; i < num_classes; ++i)
      model_file << ' ' << model->num_svs[i];
    model_file << '\n';
  }

  if (model->sv_indices) {
    model_file << "SV_indices\n";
    for (int i = 0; i < total_sv; ++i)
      model_file << model->sv_indices[i] << ' ';
    model_file << '\n';
  }

  model_file << "SVs\n";
  const double *const *tau = model->tau;
  const Node *const *svs = model->svs;

  for (int i = 0; i < total_sv; ++i) {
    for (int j = 0; j < num_classes; ++j)
      model_file << std::setprecision(16) << (tau[j][i]+0.0) << ' ';  // add "+0.0" to avoid negative zero in output

    const Node *p = svs[i];

    if (param.kernel_param->kernel_type == PRECOMPUTED) {
      model_file << "0:" << static_cast<int>(p->value) << ' ';
    } else {
      while (p->index != -1) {
        model_file << p->index << ':' << std::setprecision(8) << p->value << ' ';
        ++p;
      }
    }
    model_file << '\n';
  }

  return 0;
}

MCSVMModel *LoadMCSVMModel(std::ifstream &model_file) {
  MCSVMModel *model = new MCSVMModel;
  MCSVMParameter &param = model->param;
  param.kernel_param = new KernelParameter;
  model->sv_indices = NULL;
  model->labels = NULL;
  model->num_svs = NULL;
  model->svs = NULL;
  model->tau = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "redopt_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kRedOptTypeTable[i]; ++i) {
        if (std::strcmp(kRedOptTypeTable[i], cmd) == 0) {
          param.redopt_type = i;
          break;
        }
      }
      if (kRedOptTypeTable[i] == NULL) {
        std::cerr << "Unknown reduced optimization type.\n" << std::endl;
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "kernel_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kKernelTypeTable[i]; ++i) {
        if (std::strcmp(kKernelTypeTable[i], cmd) == 0) {
          param.kernel_param->kernel_type = i;
          break;
        }
      }
      if (kKernelTypeTable[i] == NULL) {
        std::cerr << "Unknown kernel function.\n" << std::endl;
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "degree") == 0) {
      model_file >> param.kernel_param->degree;
    } else
    if (std::strcmp(cmd, "gamma") == 0) {
      model_file >> param.kernel_param->gamma;
    } else
    if (std::strcmp(cmd, "coef0") == 0) {
      model_file >> param.kernel_param->coef0;
    } else
    if (std::strcmp(cmd, "num_examples") == 0) {
      model_file >> model->num_ex;
    } else
    if (std::strcmp(cmd, "num_classes") == 0) {
      model_file >> model->num_classes;
    } else
    if (std::strcmp(cmd, "total_SV") == 0) {
      model_file >> model->total_sv;
    } else
    if (std::strcmp(cmd, "labels") == 0) {
      int n = model->num_classes;
      model->labels = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->labels[i];
      }
    } else
    if (std::strcmp(cmd, "num_SVs") == 0) {
      int n = model->num_classes;
      model->num_svs = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->num_svs[i];
      }
    } else
    if (std::strcmp(cmd, "SV_indices") == 0) {
      int n = model->total_sv;
      model->sv_indices = new int[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->sv_indices[i];
      }
    } else
    if (std::strcmp(cmd, "SVs") == 0) {
      std::size_t n = static_cast<unsigned long>(model->num_classes);
      int total_sv = model->total_sv;
      std::string line;

      if (model_file.peek() == '\n')
        model_file.get();

      model->tau = new double*[n];
      for (int i = 0; i < n; ++i) {
        model->tau[i] = new double[total_sv];
      }
      model->svs = new Node*[total_sv];
      for (int i = 0; i < total_sv; ++i) {
        std::vector<std::string> tokens;
        std::size_t prev = 0, pos;

        std::getline(model_file, line);
        while ((pos = line.find_first_of(" \t\n", prev)) != std::string::npos) {
          if (pos > prev)
            tokens.push_back(line.substr(prev, pos-prev));
          prev = pos + 1;
        }
        if (prev < line.length())
          tokens.push_back(line.substr(prev, std::string::npos));

        for (std::size_t j = 0; j < n; ++j) {
          try
          {
            std::size_t end;
            model->tau[j][i] = std::stod(tokens[j], &end);
            if (end != tokens[j].length()) {
              throw std::invalid_argument("incomplete convention");
            }
          }
          catch(std::exception& e)
          {
            std::cerr << "Error: " << e.what() << " in SV " << (i+1) << std::endl;
            FreeMCSVMModel(model);
            std::vector<std::string>(tokens).swap(tokens);
            return NULL;
          }  // TODO try not to use exception
        }

        std::size_t elements = tokens.size() - n + 1;
        model->svs[i] = new Node[elements];
        prev = 0;
        for (std::size_t j = 0; j < elements-1; ++j) {
          pos = tokens[j+n].find_first_of(':');
          try
          {
            std::size_t end;

            model->svs[i][j].index = std::stoi(tokens[j+n].substr(prev, pos-prev), &end);
            if (end != (tokens[j+n].substr(prev, pos-prev)).length()) {
              throw std::invalid_argument("incomplete convention");
            }
            model->svs[i][j].value = std::stod(tokens[j+n].substr(pos+1), &end);
            if (end != (tokens[j+n].substr(pos+1)).length()) {
              throw std::invalid_argument("incomplete convention");
            }
          }
          catch(std::exception& e)
          {
            std::cerr << "Error: " << e.what() << " in SV " << (i+1) << std::endl;
            FreeMCSVMModel(model);
            std::vector<std::string>(tokens).swap(tokens);
            return NULL;
          }
        }
        model->svs[i][elements-1].index = -1;
        model->svs[i][elements-1].value = 0;
      }
      break;
    } else {
      std::cerr << "Unknown text in mcsvm_model file: " << cmd << std::endl;
      FreeMCSVMModel(model);
      return NULL;
    }
  }

  return model;
}

void FreeMCSVMModel(struct MCSVMModel *model) {
  if (model->svs != NULL) {
    delete[] model->svs;
    model->svs = NULL;
  }

  if (model->tau != NULL) {
    for (int i = 0; i < model->num_classes; ++i) {
      if (model->tau[i] != NULL) {
        delete[] model->tau[i];
      }
    }
    delete[] model->tau;
    model->tau = NULL;
  }

  if (model->labels != NULL) {
    delete[] model->labels;
    model->labels= NULL;
  }

  if (model->sv_indices != NULL) {
    delete[] model->sv_indices;
    model->sv_indices = NULL;
  }

  if (model->num_svs != NULL) {
    delete[] model->num_svs;
    model->num_svs = NULL;
  }

  if (model != NULL) {
    delete model;
    model = NULL;
  }

  return;
}

void FreeMCSVMParam(struct MCSVMParameter *param) {
  if (param->kernel_param != NULL) {
    delete param->kernel_param;
    param->kernel_param = NULL;
  }
  delete param;
  param = NULL;

  return;
}

void InitMCSVMParam(struct MCSVMParameter *param) {
  InitKernelParam(param->kernel_param);
  param->redopt_type = EXACT;
  param->beta = 1e-4;
  param->cache_size = 100;
  param->epsilon = 1e-3;
  param->epsilon0 = 1-1e-6;
  param->delta = 1e-4;

  return;
}

const char *CheckMCSVMParameter(const struct MCSVMParameter *param) {
  if (param->kernel_param == NULL) {
    return "no kernel parameter";
  } else if (CheckKernelParameter(param->kernel_param) != NULL) {
    return CheckKernelParameter(param->kernel_param);
  }

  int redopt_type = param->redopt_type;
  if (redopt_type != EXACT &&
      redopt_type != APPROX &&
      redopt_type != BINARY) {
    return "unknown reduced optimization type";
  }

  if (param->cache_size <= 0)
    return "cache_size <= 0";

  if (param->epsilon <= 0)
    return "epsilon <= 0";

  if (param->epsilon0 <= 0)
    return "epsilon0 <= 0";

  return NULL;
}