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


static MCDataUnit **x   = NULL;
static long *y       = NULL;
static long m        = 0;
static long k        = 0;
static long l        = 0;
static double **tau = NULL;
static double beta  = 0;

static SPOCParamDef spoc_pd;
static MCDataDef mc_datadef;
static MCSolution mc_sol;
static RedOptDef redopt_def;
static REDOPT_FUN redopt_fun;
static KF kernel_function;

static double *vector_a            = NULL;
static double **matrix_f           = NULL;
static double *row_matrix_f_next_p = NULL;
static long    **matrix_eye         = NULL;
static double *delta_tau           = NULL;
static double *old_tau             = NULL;
static double *vector_b            = NULL;

static double max_psi              =0;
static long    next_p;
static long    next_p_list;

static long    *supp_pattern_list = NULL;
static long    *zero_pattern_list = NULL;
static long    n_supp_pattern     = 0;
static long    n_zero_pattern     = 0;


int    longcmp(const void *n1, const void *n2);
void   spoc_construct();
long   spoc_epsilon(double epsilon);
void   choose_next_pattern(long *pattern_list, long n_pattern);
void   update_matrix_f(double *kernel_pattern_p);
long   allocate_memory();
void   free_auxilary_memory();
void   free_external_memory();
void   spoc_initialize();
void   update_matrix_f(double *kernel_next_p);
double next_epsilon1(double epsilon_cur, double epsilon);
double next_epsilon2(double epsilon_cur, double epsilon);
long   get_no_supp1();
double get_train_error(double b);
void   dump_matrix_f();

int   intcmp(const void *n1, const void *n2) {
  return (*(int*)(n1) - *(int*)(n2));
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
  RedOpt(int num_classes, int redopt_type);
  virtual ~RedOpt();

 protected:
  double (RedOpt::*redopt_function)();

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
  int GetMarginError(const double beta);
};

RedOpt::RedOpt(int num_classes, int redopt_type)
    :num_classes_(num_classes),
     redopt_type_(redopt_type) {
  switch (redopt_type) {
    case EXACT: {
      redopt_function = &RedOpt::RedOptExact;
      break;
    }
    case APPROX: {
      redopt_function = &RedOpt::RedOptApprox;
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
  vector_d_ = new Node[num_classes_];
}

RedOpt::~RedOpt() {
  delete[] vector_d_;
}

/* solve reduced exactly, use sort */
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




// calculate kernels
void spoc_initialize() {

  long i;
  long s,r;

  /* vector_a */
  for (i=0; i<m; i++) {
    vector_a[i] = kernel_function(x[i], x[i]);
  }

  /* matrix_eye */
  for (r=0; r<k; r++)
    for (s=0; s<k; s++)
      if (r != s)
        matrix_eye[r][s] = 0;
      else
        matrix_eye[r][s] = 1;

  /* matrix_f */
  for (i=0; i<m; i++) {
    for (r=0; r<k; r++) {
      if (y[i] != r)
        matrix_f[i][r] = 0;
      else
        matrix_f[i][r] = -beta;
    }
  }

  /* tau */
  for (i=0; i<m; i++)
    for (r=0 ;r<k; r++)
      tau[i][r] = 0;

/*  assume tau=0 */
/*  ============ */
/*    for (j=0; j<m; j++) { */
/*      for (r=0 ;r<k; r++) { */
/*        if (tau[j][r] != 0) { */
/*    for (i=0; i<m; i++) */
/*      matrix_f[i][r] += kernel_function(x[i], x[j]) * tau[j][r]; */
/*        } */
/*      } */
/*    } */

/*    for (i=0; i<m; i++) { */
/*      is_tau_r_zero = 1; */

/*      for (r=0; r<k; r++) { */
/*        if (tau[i][r] != 0) { */
/*    is_tau_r_zero = 0; */
/*    break; */
/*        } */
/*      } */
/*      if (is_tau_r_zero) */
/*        zero_pattern_list[n_zero_pattern++] =i; */
/*      else */
/*        supp_pattern_list[n_supp_pattern++] =i; */
/*    } */

  supp_pattern_list[0] =0;
  n_supp_pattern =1;

  for (i=1; i<m; i++)
    zero_pattern_list[i-1] =i;
  n_zero_pattern = m-1;
  choose_next_pattern(supp_pattern_list, n_supp_pattern);
}

long allocate_memory() {
  long i;

  /* tau */
  tau = (double **) ut_calloc(m, sizeof(double*));
  *tau = (double *) ut_calloc(m*k, sizeof(double));
  for (i=1; i<m; i++)
    tau[i] = tau[i-1] + k;

  /* vector_a */
  vector_a = (double *) ut_calloc(m, sizeof(double));

  /* matrix_f */
  matrix_f = (double **) ut_calloc(m, sizeof(double*));
  *matrix_f = (double *) ut_calloc(m*k, sizeof(double));
  for (i=1; i<m; i++)
    matrix_f[i] = matrix_f[i-1] + k;

  /* matrix_eye */
  matrix_eye = (long **) ut_calloc(k, sizeof(long*));
  *matrix_eye = (long *) ut_calloc(k*k, sizeof(long));
  for (i=1; i<k; i++)
    matrix_eye[i] = matrix_eye[i-1] + k;

  /* delta_tau */
  delta_tau = (double *) ut_calloc(k, sizeof(double));

  /* old_tau */
  old_tau = (double *) ut_calloc(k, sizeof(double));

  /* vector_b */
  vector_b = (double *) ut_calloc(k, sizeof(double));

  /* supp_pattern_list */
  supp_pattern_list = (long *) ut_calloc(m, sizeof(long));

  /* zero_pattern_list */
  zero_pattern_list = (long *) ut_calloc(m, sizeof(long));

 return (1);

}







MCSVMModel *TrainMCSVM(const struct Problem *prob, const struct MCSVMParameter *param) {
  double epsilon_current = 1;

  // spoc_construct()
  long req_n_blocks_lrucache;
  long n_blocks_lrucache;

  printf("\nOptimizer (SPOC)  (version 1.0)\n");
  printf("Initializing ... start \n"); fflush(stdout);

  x = prob.x;
  y = prob.y;
  m = prob.m;
  k = prob.k;
  l = prob.l;

  /* scale beta with m */
  fflush(stdout);
  beta = param->beta;

  printf("Requested margin (beta) %e\n", param->beta);
  kernel_function = kernel_get_function(param->kernel_def);

  allocate_memory();
  redopt_construct(k);
  kernel_construct(l);

  n_supp_pattern = 0;
  n_zero_pattern = 0;

  req_n_blocks_lrucache = MIN(m, ((long)(((double)param->cache_size) / ((((double)sizeof(double)) * ((double)m)) / ((double)MB)))));
  printf("Requesting %ld blocks of cache (External bound %ldMb)\n",
   req_n_blocks_lrucache, param->cache_size);
  n_blocks_lrucache = cachelru_construct(m, req_n_blocks_lrucache, sizeof(double)*m);

  spoc_initialize();

  printf("Initializing ... done\n"); fflush(stdout);

  // spoc_construct() end


  printf("Epsilon decreasing from %e to %e\n", param->epsilon0, param->epsilon);
  redopt_construct(k);
  redopt_fun = redopt_get_function(param->redopt_type);
  if (param->redopt_type == APPROX) {
    std::cout << std::scientific << "Delta " << param->delta << std::defaultfloat << std::endl;
  }

  redopt_def.delta = param->delta;
  redopt_def.b = (double*) ut_calloc(k, sizeof(double));

  epsilon_current = param->epsilon0;

  std::cout << "\n"
            << "New Epsilon   No. SPS      Max Psi   Train Error   Margin Error\n"
            << "-----------   -------      -------   -----------   ------------\n"
            << std::endl;

  while (max_psi > param->epsilon * beta) {
    std::cout << std::setw(11) << std::setprecision(5) << std::scientific << epsilon_current << "   "
              << std::setw(7) << std::scientific << n_supp_pattern << "   "
              << std::setw(10) << std::setprecision(3) << std::scientific << max_psi/beta << "   "
              << std::setw(7) << std::setprecision(2) << std::fixed << get_train_error(beta) << "\%      "
              << std::setw(7) << std::setprecision(2) << std::fixed << get_train_error(0) << '%'
              << std::endl;
    spoc_epsilon(epsilon_current);
    epsilon_current = next_epsilon2(epsilon_current , param->epsilon);
  }
  std::cout << std::setw(11) << std::setprecision(5) << std::scientific << param->epsilon << "   "
            << std::setw(7) << std::scientific << n_supp_pattern << "   "
            << std::setw(10) << std::setprecision(3) << std::scientific << max_psi/beta << "   "
            << std::setw(7) << std::setprecision(2) << std::fixed << get_train_error(beta) << "\%      "
            << std::setw(7) << std::setprecision(2) << std::fixed << get_train_error(0) << '%'
            << std::endl;
  free (redopt_def.b);

  {
    long i,r;

    qsort(supp_pattern_list, n_supp_pattern, sizeof(long), &longcmp);

    for (i=0; i<n_supp_pattern; i++)
      for (r=0; r<k; r++)
  tau[i][r] = tau[supp_pattern_list[i]][r];
    for (i=n_supp_pattern; i<m; i++)
      for (r=0; r<k; r++)
  tau[i][r]=0;

  }
  mc_sol.size              = m;
  mc_sol.k                 = k;
  mc_sol.l                 = l;
  mc_sol.n_supp_pattern    = n_supp_pattern;
  mc_sol.is_voted          = 0;
  mc_sol.supp_pattern_list = supp_pattern_list;
  mc_sol.votes_weight      = NULL;
  mc_sol.tau               = tau;

  std::cout << "\nNo. support pattern " << n_supp_pattern << "( " << get_no_supp1() << " at bound )\n";

  return (mc_sol);
}

long spoc_epsilon(double epsilon) {

  long supp_only =1;
  long cont = 1;
  long mistake_k;
  double *kernel_next_p;
  long r;
  long i;

  while (cont) {
    max_psi = 0;
    if (supp_only)
      choose_next_pattern(supp_pattern_list, n_supp_pattern);
    else
      choose_next_pattern(zero_pattern_list, n_zero_pattern);

    if (max_psi > epsilon * beta) {
      redopt_def.a = vector_a[next_p];
      for (r=0; r<k; r++)
        redopt_def.b[r] = matrix_f[next_p][r] - redopt_def.a * tau[next_p][r];
      redopt_def.y = y[next_p];
      for (r=0; r<k; r++)
        old_tau[r] = tau[next_p][r];
      redopt_def.alpha = tau[next_p];

      mistake_k = (*redopt_fun)(&redopt_def);

      for (r=0; r<k; r++)
        delta_tau[r] = tau[next_p][r]- old_tau[r];

      if (!cachelru_retrive(next_p, (void*)(&kernel_next_p))) {
        for (i=0; i<m; i++)
          kernel_next_p[i] = kernel_function(x[i], x[next_p]);
      }

      update_matrix_f(kernel_next_p);

      if (supp_only) {
        for (r=0; r<k; r++)
          if (tau[next_p][r] != 0) break;
        if (r == k) {
          zero_pattern_list[n_zero_pattern++] = next_p;
          supp_pattern_list[next_p_list] = supp_pattern_list[--n_supp_pattern];
        }
      } else {
        supp_pattern_list[n_supp_pattern++] = next_p;
        zero_pattern_list[next_p_list] = zero_pattern_list[--n_zero_pattern];
        supp_only =1;
      }
    } else {
      if (supp_only)
        supp_only =0;
      else
        cont =0;
    }
  }

  return (1);
}

double next_epsilon2(double epsilon_cur, double epsilon) {
  static double iteration =12;
  double e = epsilon_cur / log10(iteration);

  iteration+=2;

  return (MAX( e , epsilon));
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