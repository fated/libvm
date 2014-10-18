#include "mcsvm.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

typedef float Qfloat;
typedef signed char schar;

int CompareNodes(const void *n1, const void *n2) {
  if (((struct Node *)n1)->value > ((struct Node *)n2)->value)
    return (-1);
  else if (((struct Node *)n1)->value < ((struct Node *)n2)->value)
    return (1);
  else
    return (0);
}

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache {
 public:
  Cache(int l, long int size);
  ~Cache();

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  int get_data(const int index, Qfloat **data, int len);
  void SwapIndex(int i, int j);

 private:
  int l_;
  long int size_;
  struct Head {
    Head *prev, *next;  // a circular list
    Qfloat *data;
    int len;  // data[0,len) is cached in this entry
  };

  Head *head_;
  Head lru_head_;
  void DeleteLRU(Head *h);
  void InsertLRU(Head *h);
};

Cache::Cache(int l, long int size) : l_(l), size_(size) {
  head_ = (Head *)calloc(static_cast<size_t>(l_), sizeof(Head));  // initialized to 0
  size_ /= sizeof(Qfloat);
  size_ -= static_cast<unsigned long>(l_) * sizeof(Head) / sizeof(Qfloat);
  size_ = std::max(size_, 2 * static_cast<long int>(l_));  // cache must be large enough for two columns
  lru_head_.next = lru_head_.prev = &lru_head_;
}

Cache::~Cache() {
  for (Head *h = lru_head_.next; h != &lru_head_; h=h->next) {
    delete[] h->data;
  }
  delete[] head_;
}

void Cache::DeleteLRU(Head *h) {
  // delete from current location
  h->prev->next = h->next;
  h->next->prev = h->prev;
}

void Cache::InsertLRU(Head *h) {
  // insert to last position
  h->next = &lru_head_;
  h->prev = lru_head_.prev;
  h->prev->next = h;
  h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len) {
  Head *h = &head_[index];
  if (h->len) {
    DeleteLRU(h);
  }
  int more = len - h->len;

  if (more > 0) {
    // free old space
    while (size_ < more) {
      Head *old = lru_head_.next;
      DeleteLRU(old);
      delete[] old->data;
      size_ += old->len;
      old->data = 0;
      old->len = 0;
    }

    // allocate new space
    h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat)*static_cast<unsigned long>(len));
    size_ -= more;
    std::swap(h->len, len);
  }

  InsertLRU(h);
  *data = h->data;

  return len;
}

void Cache::SwapIndex(int i, int j) {
  if (i == j) {
    return;
  }

  if (head_[i].len) {
    DeleteLRU(&head_[i]);
  }
  if (head_[j].len) {
    DeleteLRU(&head_[j]);
  }
  std::swap(head_[i].data, head_[j].data);
  std::swap(head_[i].len, head_[j].len);
  if (head_[i].len) {
    InsertLRU(&head_[i]);
  }
  if (head_[j].len) {
    InsertLRU(&head_[j]);
  }

  if (i > j) {
    std::swap(i, j);
  }
  for (Head *h = lru_head_.next; h != &lru_head_; h = h->next) {
    if (h->len > i) {
      if (h->len > j) {
        std::swap(h->data[i], h->data[j]);
      } else {
        // give up
        DeleteLRU(h);
        delete[] h->data;
        size_ += h->len;
        h->data = 0;
        h->len = 0;
      }
    }
  }
}

// Cache end

//
// Kernel evaluation
//
// the static method KernelFunction is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
 public:
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void SwapIndex(int i, int j) const = 0;
  virtual ~QMatrix() {}
};

class Kernel : public QMatrix {
 public:
  Kernel(int l, Node *const *x, const SVMParameter& param);
  virtual ~Kernel();
  static double KernelFunction(const Node *x, const Node *y, const SVMParameter& param);
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void SwapIndex(int i, int j) const {
    std::swap(x_[i], x_[j]);
    if (x_square_) {
      std::swap(x_square_[i], x_square_[j]);
    }
  }

 protected:
  double (Kernel::*kernel_function)(int i, int j) const;

 private:
  const Node **x_;
  double *x_square_;

  // SVMParameter
  const int kernel_type;
  const int degree;
  const double gamma;
  const double coef0;

  static double Dot(const Node *px, const Node *py);
  double KernelLinear(int i, int j) const {
    return Dot(x_[i], x_[j]);
  }
  double KernelPoly(int i, int j) const {
    return std::pow(gamma*Dot(x_[i], x_[j])+coef0, degree);
  }
  double KernelRBF(int i, int j) const {
    return exp(-gamma*(x_square_[i]+x_square_[j]-2*Dot(x_[i], x_[j])));
  }
  double KernelSigmoid(int i, int j) const {
    return tanh(gamma*Dot(x_[i], x_[j])+coef0);
  }
  double KernelPrecomputed(int i, int j) const {
    return x_[i][static_cast<int>(x_[j][0].value)].value;
  }
};

Kernel::Kernel(int l, Node *const *x, const SVMParameter &param)
    :kernel_type(param.kernel_type),
     degree(param.degree),
     gamma(param.gamma),
     coef0(param.coef0) {
  switch (kernel_type) {
    case LINEAR: {
      kernel_function = &Kernel::KernelLinear;
      break;
    }
    case POLY: {
      kernel_function = &Kernel::KernelPoly;
      break;
    }
    case RBF: {
      kernel_function = &Kernel::KernelRBF;
      break;
    }
    case SIGMOID: {
      kernel_function = &Kernel::KernelSigmoid;
      break;
    }
    case PRECOMPUTED: {
      kernel_function = &Kernel::KernelPrecomputed;
      break;
    }
    default: {
      // assert(false);
      break;
    }
  }

  clone(x_, x, l);

  if (kernel_type == RBF) {
    x_square_ = new double[l];
    for (int i = 0; i < l; ++i) {
      x_square_[i] = Dot(x_[i], x_[i]);
    }
  } else {
    x_square_ = 0;
  }
}

Kernel::~Kernel() {
  delete[] x_;
  delete[] x_square_;
}

double Kernel::Dot(const Node *px, const Node *py) {
  double sum = 0;
  while (px->index != -1 && py->index != -1) {
    if (px->index == py->index) {
      sum += px->value * py->value;
      ++px;
      ++py;
    } else {
      if (px->index > py->index) {
        ++py;
      } else {
        ++px;
      }
    }
  }

  return sum;
}

double Kernel::KernelFunction(const Node *x, const Node *y, const SVMParameter &param) {
  switch (param.kernel_type) {
    case LINEAR: {
      return Dot(x, y);
    }
    case POLY: {
      return std::pow(param.gamma*Dot(x, y)+param.coef0, param.degree);
    }
    case RBF: {
      double sum = 0;
      while (x->index != -1 && y->index != -1) {
        if (x->index == y->index) {
          double d = x->value - y->value;
          sum += d*d;
          ++x;
          ++y;
        } else {
          if (x->index > y->index) {
            sum += y->value * y->value;
            ++y;
          } else {
            sum += x->value * x->value;
            ++x;
          }
        }
      }

      while (x->index != -1) {
        sum += x->value * x->value;
        ++x;
      }

      while (y->index != -1) {
        sum += y->value * y->value;
        ++y;
      }

      return exp(-param.gamma*sum);
    }
    case SIGMOID: {
      return tanh(param.gamma*Dot(x, y)+param.coef0);
    }
    case PRECOMPUTED: {  //x: test (validation), y: SV
      return x[static_cast<int>(y->value)].value;
    }
    default: {
      // assert(false);
      return 0;  // Unreachable
    }
  }
}

// Kernel end

// ReducedOptimization start

class RedOpt {
 public:
  RedOpt(int num_classes, MCSVMParameter *param);
  virtual ~RedOpt();
  void set_a(double a) {
    a_ = a;
  }
  double get_a() {
    return a_;
  }
  void set_y(double y) {
    y_ = y;
  }
  int get_y() {
    return y_;
  }
  void set_b(double b, int r) {
    b_[r] = b;
  }
  double get_b(int r) {
    return b_[r];
  }

 protected:
  double (RedOpt::*redopt_function)();
  int GetMarginError(const double beta);

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

  static void Two(double v0, double v1, int i0, int i1) {
    double temp = 0.5 * (v0-v1);
    temp = (temp < 1) ? temp : 1;
    alpha_[i0] = -temp;
    alpha_[i1] = temp;
  }
  int RedOptExact();
  int RedOptApprox();
  int RedOptAnalyticBinary();
};

RedOpt::RedOpt(int num_classes, MCSVMParameter *param)
    :num_classes_(num_classes),
     redopt_type_(param->redopt_type)
     delta_(param->delta) {
  switch (redopt_type) {
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
}

RedOpt::~RedOpt() {
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
  for (r = 0; r < num_classes_; ++r) {
    if (b_[r] > b_[y_]) {
      vector_d_[mistake_k].index = r;
      vector_d_[mistake_k].value = b_[r] / a_;
      sum_d += vector_d_[mistake_k].value;
      ++mistake_k;
    } else {  // for other labels, alpha=0
      alpha_[r] = 0;
    }
  }

  /* if no mistake labels return */
  if (mistake_k == 0) {
    return 0;
  }
  /* add correct label to list (up to constant one) */
  vector_d_[mistake_k].index = y_;
  vector_d_[mistake_k].value = b_[y_] / a_;

  /* if there are only two bad labels, solve for it */
  if (mistake_k == 1) {
    Two(vector_d_[0].value, vector_d_[1].value, vector_d_[0].index, vector_d_[1].index);
    return 2;
  }

  /* finish calculations */
  sum_d += vector_d_[mistake_k].value;
  ++vector_d_[mistake_k].value;
  ++mistake_k;

  /* sort vector_d by values */
  qsort(vector_d_, mistake_k, sizeof(struct Node), CompareNodes);

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
  }
  /* theta > min vector_d.value */
  else {
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
    alpha_[y_]++;
  }

  return (mistake_k);
}

int RedOpt::RedOptApprox() {

  double old_theta = DBL_MAX;  /* threshold */
  double theta = DBL_MAX;      /* threshold */
  double temp;
  int mistake_k =0; /* no. of labels with score greater than the correct label */
  int r;

  /* pick only problematic labels */
  for (r = 0; r < k; ++r) {
    if (b_[r] > b_[y_]) {
      vector_d_[mistake_k].index = r;
      vector_d_[mistake_k].value = b_[r] / a_;
      ++mistake_k;
    }
    /* for other labels, alpha=0 */
    else {
      alpha_[r] = 0;
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
  for (r = 0; r < mistake_k; ++r) {
    if (vector_d_[r].value < theta)
      theta = vector_d_[r].value;
  }

  /* loop until convergence of theta */
  while (1) {
    old_theta = theta;

    /* calculate new value of theta */
    theta = -1;
    for (r = 0; r < mistake_k; ++r) {
      if (old_theta > vector_d_[r].value) {
        theta += old_theta;
      } else {
        theta += vector_d_[r].value;
      }
    }
    theta /= mistake_k;

    if (fabs((old_theta-theta)/theta) < rod->delta) {
      break;
    }
  }

  /* update alpha using threshold */
  for (r = 0; r < mistake_k; ++r) {
    temp = theta - vector_d_[r].value;
    if (temp < 0) {
      alpha_[vector_d_[r].index] = temp;
    } else {
      alpha_[vector_d_[r].index] = 0;
    }
  }
  ++alpha_[y_];

  return(mistake_k);
}

/* solve for k=2 */
int RedOpt::RedOptAnalyticBinary() {
  int y0 = 1 - y_; /* other label */
  int y1 = y_;    /* currect label */

  if (b_[y0] > b_[y1]) {
    vector_d_[y0].value = b_[y0] / a_;
    vector_d_[y1].value = b_[y1] / a_;

    Two(vector_d_[y0].value, vector_d_[y1].value, y0, y1);
    return (2);
  } else {
    alpha_[0] = alpha_[1] = 0;
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

// RedOpt start

// Spoc start

class Spoc {
 public:
  Spoc() {};
  virtual ~Spoc() {};
  void Solve();
  double get_max_psi() {
    return max_psi_;
  }
  double get_beta() {
    return beta_;
  }

 protected:
  RedOpt red_opt_;

 private:
  static int iteration_ = 12;
  int num_ex_;
  int num_classes_;
  int cache_size_;
  int num_support_pattern_;
  int num_zero_pattern_;
  int next_p_;
  int next_p_list_;
  int *support_pattern_list;
  int *zero_pattern_list;
  int **matrix_eye_;
  double max_psi_;
  double *row_matrix_f_next_p_;
  double *vector_a_;
  double *vector_b_;
  double beta_;
  double epsilon_;
  double epsilon0_;
  double delta_;
  double *delta_tau_;
  double *old_tau_;
  double **matrix_f_;
  double **tau_;
  SPOC_Q spoc_Q_;
};

Spoc::Spoc(Problem *prob, MCSVMParameter &param)
    :beta_(param->beta)
     num_ex_(prob->num_ex)
     num_classes_(param->num_classes)
     cache_size_(param->cache_size) {
  long req_n_blocks_lrucache;
  long n_blocks_lrucache;

  Info("\nOptimizer (SPOC)  (version 1.0)\n");
  Info("Initializing ... start\n");

  Info("Requested margin (beta) %e\n", beta_);

  kernel_function = kernel_get_function(spoc_pd.kernel_def);

  // allocate memory start
  // tau
  tau_ = new double*[num_ex_];
  *tau = new double[num_ex_ * num_classes_];
  for (int i = 1; i < num_ex_; ++i) {
    tau_[i] = tau_[i-1] + num_classes_;
  }

  // vector_a
  vector_a_ = new double[num_ex_];

  // matrix_f
  matrix_f_ = new double*[num_ex_];
  *matrix_f_ = new double[num_ex_ * num_classes_];
  for (int i = 1; i < num_ex_; ++i)
    matrix_f_[i] = matrix_f_[i-1] + num_classes_;

  // matrix_eye
  matrix_eye_ = new int*[num_classes_];
  *matrix_eye_ = new int[num_classes_ * num_classes_];
  for (int i = 1; i < num_classes_; ++i)
    matrix_eye_[i] = matrix_eye_[i-1] + num_classes_;

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

  RedOpt red_opt_ = new RedOpt(num_classes_, param);

  num_support_pattern_ = 0;
  num_zero_pattern_ = 0;

  req_n_blocks_lrucache = std::min(m, ((long)(((double)cache_size_) / ((sizeof(double) * ((double)num_ex_)) / MB))));
  Info("Requesting %ld blocks of cache (External bound %ldMb)\n", req_n_blocks_lrucache, cache_size_);
  spoc_Q = new SPOC_Q(prob, param);

  // initialize begins

  // vector_a
  double *QD = spoc_Q.get_QD();
  for (int i = 0; i < num_ex_; ++i) {
    vector_a_[i] = QD[i];
  }

  // matrix_eye
  for (int r = 0; r < num_classes_; ++r) {
    for (int s = 0; s < num_classes_; ++s) {
      if (r != s) {
        matrix_eye_[r][s] = 0;
      } else {
        matrix_eye_[r][s] = 1;
      }
    }
  }

  // matrix_f
  for (int i = 0; i < num_ex_; ++i) {
    for (int r = 0; r < num_classes_; ++r) {
      if (y[i] != r) {
        matrix_f_[i][r] = 0;
      } else {
        matrix_f_[i][r] = -beta_;
      }
    }
  }

  // tau
  for (int i = 0; i < num_ex_; ++i) {
    for (int r = 0 ; r < num_classes_; ++r) {
      tau_[i][r] = 0;
    }
  }

  support_pattern_list_[0] = 0;
  num_support_pattern_ = 1;

  for (int i = 1; i < num_ex_; ++i) {
    zero_pattern_list_[i-1] = i;
  }
  num_zero_pattern_ = num_ex_-1;
  ChooseNextPattern(support_pattern_list_, num_support_pattern_);

  // initialize ends

  Info("Initializing ... done\n");
}

Spoc::~Spoc() {
  if (vector_a_ != NULL) {
    delete[] vector_a_;
  }
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
}

void Spoc::Solve(double epsilon) {
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
      redopt_def.a = vector_a_[next_p_];
      for (int r = 0; r < num_classes_; ++r) {
        redopt_def.b[r] = matrix_f_[next_p_][r] - redopt_def.a * tau_[next_p_][r];
      }
      redopt_def.y = y[next_p_];
      for (int r = 0; r < num_classes_; ++r) {
        old_tau_[r] = tau_[next_p_][r];
      }
      redopt_def.alpha = tau_[next_p_];

      mistake_k = red_opt_->*redopt_function();

      for (int r = 0; r < num_classes_; ++r) {
        delta_tau_[r] = tau_[next_p_][r] - old_tau_[r];
      }

      if (!cachelru_retrive(next_p, (void*)(&kernel_next_p))) {
        for (int i = 0; i < num_ex_; ++i)
          kernel_next_p_[i] = spoc_Q->*kernel_function(i, next_p_);
      }

      UpdateMatrix(kernel_next_p);

      if (supp_only) {
        int r;
        for (r = 0; r < num_classes_; ++r) {
          if (tau_[next_p_][r] != 0) {
            break;
          }
        }
        if (r == k) {
          zero_pattern_list_[num_zero_pattern_++] = next_p_;
          support_pattern_list_[next_p_list_] = support_pattern_list_[--num_support_pattern_];
        }
      } else {
        support_pattern_list_[num_support_pattern__++] = next_p_;
        zero_pattern_list[next_p_list_] = zero_pattern_list_[--num_zero_pattern_];
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

double Spoc::NextEpsilon(double epsilon_cur, double epsilon) {
  double e = epsilon_cur / log10(iteration_);
  iteration_ += 2;

  return (std::max(e, epsilon));
}

void Spoc::ChooseNextPattern(int *pattern_list, int num_patterns) {
  // psi : KKT value of example
  // psi1 : max_r matrix_f[i][r]
  // psi0 : min_{r, tau[i][r]<delta[yi][r]}  matrix_f[i][r]
  int p = 0;
  double *matrix_f_ptr;

  for (int i = 0; i < num_patterns; ++i) {
    double psi1 = -DBL_MAX;
    double psi0 = DBL_MAX;

    p = pattern_list[i];
    matrix_f_ptr = matrix_f_[p];

    for (int r = 0; r < num_classes_; ++r, ++matrix_f_ptr) {
      if (*matrix_f_ptr > psi1)
        psi1 = *matrix_f_ptr;

      if (*matrix_f_ptr < psi0)
        if (tau_[p][r] < matrix_eye_[y[p]][r])
          psi0 = *matrix_f_ptr;
    }

    double psi = psi1 - psi0;

    if (psi > max_psi) {
      next_p_list_ = i;
      max_psi_ = psi;
    }
  }
  next_p_ = pattern_list[next_p_list_];
  row_matrix_f_next_p_ = matrix_f_[p];
}

void Spoc::UpdateMatrix(double *kernel_next_p) {
  double *delta_tau_ptr = delta_tau_;
  double *kernel_next_p_ptr;

  for (int r = 0; r < k; ++r, ++delta_tau_ptr) {
    if (*delta_tau_ptr != NULL) {
      for (int i = 0, kernel_next_p_ptr = kernel_next_p; i < num_ex_; ++i, ++kernel_next_p_ptr) {
        matrix_f_[i][r] += (*delta_tau_ptr) * (*kernel_next_p_ptr);
      }
    }
  }

  return;
}

double Spoc::CalcTrainError(double beta) {
  int errors = 0;

  for (int i = 0; i < num_ex_; ++i) {
    int j;
    double max = -DBL_MAX;
    for (j = 0; j < y[i]; ++j) {
      if (matrix_f_[i][j] > max) {
        max = matrix_f_[i][j];
      }
    }
    for (++j; j < num_classes_; ++j) {
      if (matrix_f_[i][j] > max) {
        max = matrix_f_[i][j];
      }
    }
    if ((max-beta) >= matrix_f_[i][y[i]]) {
      ++errors;
    }
  }

  return (100*errors/(static_cast<double>(num_ex_)));
}

int Spoc::CountNumSVs() {
  int n = 0;

  for (int i = 0; i < num_ex_; ++i)
    if (tau_[i][y[i]] == 1) {
      ++n;
    }

  return n;
}

void Spoc::PrintEpsilon(double epsilon) {
  Info("%11.5e   %7ld   %10.3e   %7.2f%%      %7.2f%%\n",
    epsilon, num_support_pattern_, max_psi_/beta_, red_opt_.GetMarginError(beta_), red_opt_.GetMarginError(0));

  return;
}

// Spoc class end

// Q class begin

class SPOC_Q : public Kernel {
 public:
  SPOC_Q(const Problem &prob, const MCSVMParameter &param) : Kernel(prob.num_ex, prob.x, param) {
    cache_ = new Cache(prob.num_ex, static_cast<long int>(param.cache_size*(1<<20)));
    QD_ = new double[prob.num_ex];
    for (int i = 0; i < prob.num_ex; ++i)
      QD_[i] = (this->*kernel_function)(i, i);
  }

  double *get_QD() const {
    return QD_;
  }

  ~SPOC_Q() {
    delete[] cache_;
    delete[] QD_;
  }

 private:
  Cache *cache_;
  double *QD_;  // Q matrix Diagonal
};

// Q class ends

MCSVMModel *TrainMCSVM(const struct Problem *prob, const struct MCSVMParameter *param) {
  double epsilon_current = 1;
  // group problem
  Spoc s = new Spoc(prob, param);

  Info("Epsilon decreasing from %e to %e\n", param->epsilon0, param->epsilon);
  epsilon_current = param->epsilon0;

  Info("\nNew Epsilon   No. SPS      Max Psi   Train Error   Margin Error\n");
  Info("-----------   -------      -------   -----------   ------------\n");

  while (s.get_max_psi() > param->epsilon * s.get_beta()) {
    s.PrintEpsilon(epsilon_current);
    s.Solve(epsilon_current);
    epsilon_current = s.NextEpsilon(epsilon_current, param->epsilon);
  }
  s.PrintEpsilon(param->epsilon);

  qsort(support_pattern_list_, num_support_pattern_, sizeof(long), &longcmp);

  for (int i = 0; i < num_support_pattern_; ++i) {
    for (int r = 0; r < num_classes_; ++r) {
      tau_[i][r] = tau_[support_pattern_list_[i]][r];
    }
  }
  for (int i = num_support_pattern_; i < num_ex_; ++i) {
    for (int r = 0; r < num_classes_; ++r) {
      tau_[i][r] = 0;
    }
  }

  mc_sol.size              = m;
  mc_sol.k                 = k;
  mc_sol.l                 = l;
  mc_sol.num_support_pattern_    = num_support_pattern_;
  mc_sol.is_voted          = 0;
  mc_sol.support_pattern_list_ = support_pattern_list_;
  mc_sol.votes_weight      = NULL;
  mc_sol.tau               = tau;

  std::cout << "\nNo. support pattern " << num_support_pattern_ << "( " << get_no_supp1() << " at bound )\n";

  return (mc_sol);
}

double PredictMCSVM(const struct MCSVMModel *model, const struct Node *x) {

}

int SaveMCSVMModel(std::ofstream &model_file, const struct MCSVMModel *model) {

}

MCSVMModel *LoadMCSVMModel(std::ifstream &model_file) {

}

void FreeMCSVMModel(struct MCSVMModel **model) {

}

void FreeMCSVMParam(struct MCSVMParameter *param) {

}

void InitMCSVMParam(struct MCSVMParameter *param) {

  param->beta = 1e-4;
  param->cache_size = 4096;

  param->kernel_type = RBF;
  param->degree = 1;
  param->coef0 = 1;
  param->gamma = 1;

  param->epsilon = 1e-3;
  param->epsilon0 = 1-1e-6;
  param->delta = 1e-4;
  param->redopt_type = EXACT;

  return;
}

const char *CheckMCSVMParameter(const struct MCSVMParameter *param) {

}