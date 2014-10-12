#include "svm.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdarg>
#include <string>
#include <vector>
#include <exception>

typedef float Qfloat;
typedef signed char schar;

static void PrintCout(const char *s) {
  std::cout << s;
  std::cout.flush();
}

static void PrintNull(const char *s) {}

static void (*PrintString) (const char *) = &PrintNull;

static void info(const char *fmt, ...) {
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap, fmt);
  vsprintf(buf, fmt, ap);
  va_end(ap);
  (*PrintString)(buf);
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
  void swap_index(int i, int j);

 private:
  int l;
  long int size;
  struct Head {
    Head *prev, *next;  // a circular list
    Qfloat *data;
    int len;    // data[0,len) is cached in this entry
  };

  Head *head;
  Head lru_head;
  void lru_delete(Head *h);
  void lru_insert(Head *h);
};

Cache::Cache(int l_, long int size_) : l(l_), size(size_) {
  head = (Head *)calloc((size_t)l, sizeof(Head));  // initialized to 0
  size /= sizeof(Qfloat);
  size -= (unsigned long)l * sizeof(Head) / sizeof(Qfloat);
  size = std::max(size, 2 * (long int) l);  // cache must be large enough for two columns
  lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache() {
  for (Head *h = lru_head.next; h != &lru_head; h=h->next)
    delete[] h->data;
  delete[] head;
}

void Cache::lru_delete(Head *h) {
  // delete from current location
  h->prev->next = h->next;
  h->next->prev = h->prev;
}

void Cache::lru_insert(Head *h) {
  // insert to last position
  h->next = &lru_head;
  h->prev = lru_head.prev;
  h->prev->next = h;
  h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len) {
  Head *h = &head[index];
  if (h->len) lru_delete(h);
  int more = len - h->len;

  if (more > 0) {
    // free old space
    while (size < more) {
      Head *old = lru_head.next;
      lru_delete(old);
      delete[] old->data;
      size += old->len;
      old->data = 0;
      old->len = 0;
    }

    // allocate new space
    h->data = (Qfloat *)realloc(h->data, sizeof(Qfloat)*(unsigned long)len);
    size -= more;
    std::swap(h->len, len);
  }

  lru_insert(h);
  *data = h->data;
  return len;
}

void Cache::swap_index(int i, int j) {
  if (i == j) return;

  if (head[i].len) lru_delete(&head[i]);
  if (head[j].len) lru_delete(&head[j]);
  std::swap(head[i].data, head[j].data);
  std::swap(head[i].len, head[j].len);
  if (head[i].len) lru_insert(&head[i]);
  if (head[j].len) lru_insert(&head[j]);

  if (i > j) std::swap(i, j);
  for (Head *h = lru_head.next; h!=&lru_head; h=h->next) {
    if (h->len > i) {
      if (h->len > j) {
        std::swap(h->data[i], h->data[j]);
      } else {
        // give up
        lru_delete(h);
        delete[] h->data;
        size += h->len;
        h->data = 0;
        h->len = 0;
      }
    }
  }
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
 public:
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void swap_index(int i, int j) const = 0;
  virtual ~QMatrix() {}
};

class Kernel : public QMatrix {
 public:
  Kernel(int l, Node *const *x, const SVMParameter& param);
  virtual ~Kernel();
  static double k_function(const Node *x, const Node *y, const SVMParameter& param);
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void swap_index(int i, int j) const {
    std::swap(x[i], x[j]);
    if (x_square) std::swap(x_square[i], x_square[j]);
  }

 protected:
  double (Kernel::*kernel_function)(int i, int j) const;

 private:
  const Node **x;
  double *x_square;

  // SVMParameter
  const int kernel_type;
  const int degree;
  const double gamma;
  const double coef0;

  static double dot(const Node *px, const Node *py);
  double kernel_linear(int i, int j) const {
    return dot(x[i], x[j]);
  }
  double kernel_poly(int i, int j) const {
    return std::pow(gamma*dot(x[i], x[j])+coef0, degree);
  }
  double kernel_rbf(int i, int j) const {
    return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i], x[j])));
  }
  double kernel_sigmoid(int i, int j) const {
    return tanh(gamma*dot(x[i], x[j])+coef0);
  }
  double kernel_precomputed(int i, int j) const {
    return x[i][(int)(x[j][0].value)].value;
  }
};

Kernel::Kernel(int l, Node *const *x_, const SVMParameter &param)
    :kernel_type(param.kernel_type),
     degree(param.degree),
     gamma(param.gamma),
     coef0(param.coef0) {
  switch (kernel_type) {
    case LINEAR: {
      kernel_function = &Kernel::kernel_linear;
      break;
    }
    case POLY: {
      kernel_function = &Kernel::kernel_poly;
      break;
    }
    case RBF: {
      kernel_function = &Kernel::kernel_rbf;
      break;
    }
    case SIGMOID: {
      kernel_function = &Kernel::kernel_sigmoid;
      break;
    }
    case PRECOMPUTED: {
      kernel_function = &Kernel::kernel_precomputed;
      break;
    }
    default: {
      // assert(false);
    }
  }

  clone(x, x_, l);

  if (kernel_type == RBF) {
    x_square = new double[l];
    for (int i = 0; i < l; ++i)
      x_square[i] = dot(x[i], x[i]);
  } else {
    x_square = 0;
  }
}

Kernel::~Kernel() {
  delete[] x;
  delete[] x_square;
}

double Kernel::dot(const Node *px, const Node *py) {
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

double Kernel::k_function(const Node *x, const Node *y, const SVMParameter &param) {
  switch (param.kernel_type) {
    case LINEAR: {
      return dot(x, y);
    }
    case POLY: {
      return std::pow(param.gamma*dot(x, y)+param.coef0, param.degree);
    }
    case RBF: {
      double sum = 0;
      while (x->index != -1 && y->index !=-1) {
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
      return tanh(param.gamma*dot(x, y)+param.coef0);
    }
    case PRECOMPUTED: {  //x: test (validation), y: SV
      return x[(int)(y->value)].value;
    }
    default: {
      // assert(false);
      return 0;  // Unreachable
    }
  }
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//  min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//    y^T \alpha = \delta
//    y_i = +1 or -1
//    0 <= alpha_i <= Cp for y_i = 1
//    0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//  Q, p, y, Cp, Cn, and an initial feasible point \alpha
//  l is the size of vectors and matrices
//  eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
 public:
  Solver() {};
  virtual ~Solver() {};

  struct SolutionInfo {
    double obj;
    double rho;
    double upper_bound_p;
    double upper_bound_n;
    double r;  // for Solver_NU
  };

  void Solve(int l, const QMatrix &Q, const double *p_, const schar *y_,
      double *alpha_, double Cp, double Cn, double eps,
      SolutionInfo *si, int shrinking);

 protected:
  int active_size;
  schar *y;
  double *G;  // gradient of objective function
  enum { LOWER_BOUND, UPPER_BOUND, FREE };
  char *alpha_status;  // LOWER_BOUND, UPPER_BOUND, FREE
  double *alpha;
  const QMatrix *Q;
  const double *QD;
  double eps;
  double Cp,Cn;
  double *p;
  int *active_set;
  double *G_bar;  // gradient, if we treat free variables as 0
  int l;
  bool unshrink;  // XXX

  double get_C(int i) {
    return (y[i] > 0) ? Cp : Cn;
  }
  void update_alpha_status(int i) {
    if (alpha[i] >= get_C(i)) {
      alpha_status[i] = UPPER_BOUND;
    } else {
      if (alpha[i] <= 0) {
        alpha_status[i] = LOWER_BOUND;
      } else {
        alpha_status[i] = FREE;
      }
    }
  }
  bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
  bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
  bool is_free(int i) { return alpha_status[i] == FREE; }
  void swap_index(int i, int j);
  void reconstruct_gradient();
  virtual int select_working_set(int &i, int &j);
  virtual double calculate_rho();
  virtual void do_shrinking();

 private:
  bool be_shrunk(int i, double Gmax1, double Gmax2);
};

void Solver::swap_index(int i, int j) {
  Q->swap_index(i, j);
  std::swap(y[i], y[j]);
  std::swap(G[i], G[j]);
  std::swap(alpha_status[i], alpha_status[j]);
  std::swap(alpha[i], alpha[j]);
  std::swap(p[i], p[j]);
  std::swap(active_set[i], active_set[j]);
  std::swap(G_bar[i], G_bar[j]);
}

void Solver::reconstruct_gradient() {
  // reconstruct inactive elements of G from G_bar and free variables
  if (active_size == l) return;

  int i, j;
  int nr_free = 0;

  for (j = active_size; j < l; ++j) {
    G[j] = G_bar[j] + p[j];
  }

  for (j = 0; j < active_size; ++j) {
    if (is_free(j)) {
      nr_free++;
    }
  }

  if (2*nr_free < active_size) {
    info("\nWARNING: using -h 0 may be faster\n");
  }

  if (nr_free*l > 2*active_size*(l-active_size)) {
    for (i = active_size; i < l; ++i) {
      const Qfloat *Q_i = Q->get_Q(i, active_size);
      for (j = 0; j < active_size; ++j) {
        if (is_free(j)) {
          G[i] += alpha[j] * Q_i[j];
        }
      }
    }
  } else {
    for (i = 0; i < active_size; ++i) {
      if (is_free(i)) {
        const Qfloat *Q_i = Q->get_Q(i, l);
        double alpha_i = alpha[i];
        for (j = active_size; j < l; ++j) {
          G[j] += alpha_i * Q_i[j];
        }
      }
    }
  }
}

void Solver::Solve(int l, const QMatrix &Q, const double *p_, const schar *y_,
    double *alpha_, double Cp, double Cn, double eps,
    SolutionInfo *si, int shrinking) {
  this->l = l;
  this->Q = &Q;
  QD=Q.get_QD();
  clone(p, p_, l);
  clone(y, y_, l);
  clone(alpha, alpha_, l);
  this->Cp = Cp;
  this->Cn = Cn;
  this->eps = eps;
  unshrink = false;

  // initialize alpha_status
  alpha_status = new char[l];
  for (int i = 0; i < l; ++i) {
    update_alpha_status(i);
  }

  // initialize active set (for shrinking)
  active_set = new int[l];
  for (int i = 0; i < l; ++i) {
    active_set[i] = i;
  }
  active_size = l;

  // initialize gradient
  G = new double[l];
  G_bar = new double[l];
  for (int i = 0; i < l; ++i) {
    G[i] = p[i];
    G_bar[i] = 0;
  }
  for (int i = 0; i < l; ++i)
    if (!is_lower_bound(i)) {
      const Qfloat *Q_i = Q.get_Q(i,l);
      double alpha_i = alpha[i];
      int j;
      for (j = 0; j < l; ++j) {
        G[j] += alpha_i*Q_i[j];
      }
      if (is_upper_bound(i)) {
        for (j = 0; j < l; ++j) {
          G_bar[j] += get_C(i) * Q_i[j];
        }
      }
    }

  // optimization step
  int iter = 0;
  int max_iter = std::max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
  int counter = std::min(l,1000)+1;

  while (iter < max_iter) {
    // show progress and do shrinking
    if (--counter == 0) {
      counter = std::min(l,1000);
      if (shrinking) do_shrinking();
      info(".");
    }

    int i,j;
    if (select_working_set(i,j) != 0) {
      // reconstruct the whole gradient
      reconstruct_gradient();
      // reset active set size and check
      active_size = l;
      info("*");
      if (select_working_set(i,j) != 0) {
        break;
      } else {
        counter = 1;  // do shrinking next iteration
      }
    }

    ++iter;

    // update alpha[i] and alpha[j], handle bounds carefully
    const Qfloat *Q_i = Q.get_Q(i, active_size);
    const Qfloat *Q_j = Q.get_Q(j, active_size);

    double C_i = get_C(i);
    double C_j = get_C(j);

    double old_alpha_i = alpha[i];
    double old_alpha_j = alpha[j];

    if (y[i] != y[j]) {
      double quad_coef = QD[i]+QD[j]+2*Q_i[j];
      if (quad_coef <= 0)
        quad_coef = TAU;
      double delta = (-G[i]-G[j])/quad_coef;
      double diff = alpha[i] - alpha[j];
      alpha[i] += delta;
      alpha[j] += delta;

      if (diff > 0) {
        if (alpha[j] < 0) {
          alpha[j] = 0;
          alpha[i] = diff;
        }
      } else {
        if (alpha[i] < 0) {
          alpha[i] = 0;
          alpha[j] = -diff;
        }
      }
      if (diff > C_i - C_j) {
        if (alpha[i] > C_i) {
          alpha[i] = C_i;
          alpha[j] = C_i - diff;
        }
      } else {
        if (alpha[j] > C_j) {
          alpha[j] = C_j;
          alpha[i] = C_j + diff;
        }
      }
    } else {
      double quad_coef = QD[i]+QD[j]-2*Q_i[j];
      if (quad_coef <= 0)
        quad_coef = TAU;
      double delta = (G[i]-G[j])/quad_coef;
      double sum = alpha[i] + alpha[j];
      alpha[i] -= delta;
      alpha[j] += delta;

      if (sum > C_i) {
        if (alpha[i] > C_i) {
          alpha[i] = C_i;
          alpha[j] = sum - C_i;
        }
      } else {
        if (alpha[j] < 0) {
          alpha[j] = 0;
          alpha[i] = sum;
        }
      }
      if (sum > C_j) {
        if (alpha[j] > C_j) {
          alpha[j] = C_j;
          alpha[i] = sum - C_j;
        }
      } else {
        if (alpha[i] < 0) {
          alpha[i] = 0;
          alpha[j] = sum;
        }
      }
    }

    // update G
    double delta_alpha_i = alpha[i] - old_alpha_i;
    double delta_alpha_j = alpha[j] - old_alpha_j;

    for (int k = 0; k < active_size; ++k) {
      G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
    }

    // update alpha_status and G_bar
    bool ui = is_upper_bound(i);
    bool uj = is_upper_bound(j);
    update_alpha_status(i);
    update_alpha_status(j);
    int k;
    if (ui != is_upper_bound(i)) {
      Q_i = Q.get_Q(i,l);
      if (ui) {
        for (k = 0; k < l; ++k)
          G_bar[k] -= C_i * Q_i[k];
      } else {
        for (k = 0; k < l; ++k)
          G_bar[k] += C_i * Q_i[k];
      }
    }

    if (uj != is_upper_bound(j)) {
      Q_j = Q.get_Q(j,l);
      if (uj) {
        for (k = 0; k < l; ++k)
          G_bar[k] -= C_j * Q_j[k];
      } else {
        for (k = 0; k < l; ++k)
          G_bar[k] += C_j * Q_j[k];
      }
    }
  }

  if (iter >= max_iter) {
    if (active_size < l) {
      // reconstruct the whole gradient to calculate objective value
      reconstruct_gradient();
      active_size = l;
      info("*");
    }
    fprintf(stderr, "\nWARNING: reaching max number of iterations\n");
  }

  // calculate rho
  si->rho = calculate_rho();

  // calculate objective value
  double v = 0;
  for (int i = 0; i < l; ++i)
    v += alpha[i] * (G[i] + p[i]);
  si->obj = v/2;

  // put back the solution
  for (int i = 0; i < l; ++i)
    alpha_[active_set[i]] = alpha[i];

  // juggle everything back
  /*{
    for(int i=0;i<l;i++)
      while(active_set[i] != i)
        swap_index(i,active_set[i]);
        // or Q.swap_index(i,active_set[i]);
  }*/

  si->upper_bound_p = Cp;
  si->upper_bound_n = Cn;

  info("\noptimization finished, #iter = %d\n", iter);

  delete[] p;
  delete[] y;
  delete[] alpha;
  delete[] alpha_status;
  delete[] active_set;
  delete[] G;
  delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j) {
  // return i,j such that
  // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  // j: minimizes the decrease of obj value
  //    (if quadratic coefficeint <= 0, replace it with tau)
  //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

  double Gmax = -INF;
  double Gmax2 = -INF;
  int Gmax_idx = -1;
  int Gmin_idx = -1;
  double obj_diff_min = INF;

  for (int t = 0; t < active_size; ++t)
    if (y[t] == +1) {
      if (!is_upper_bound(t))
        if (-G[t] >= Gmax) {
          Gmax = -G[t];
          Gmax_idx = t;
        }
    } else {
      if (!is_lower_bound(t))
        if (G[t] >= Gmax) {
          Gmax = G[t];
          Gmax_idx = t;
        }
    }

  int i = Gmax_idx;
  const Qfloat *Q_i = NULL;
  if (i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
    Q_i = Q->get_Q(i, active_size);

  for (int j = 0; j < active_size; ++j) {
    if (y[j] == +1) {
      if (!is_lower_bound(j)) {
        double grad_diff=Gmax+G[j];
        if (G[j] >= Gmax2)
          Gmax2 = G[j];
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff)/TAU;
          }

          if (obj_diff <= obj_diff_min) {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    } else {
      if (!is_upper_bound(j)) {
        double grad_diff= Gmax-G[j];
        if (-G[j] >= Gmax2)
          Gmax2 = -G[j];
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff)/TAU;
          }

          if (obj_diff <= obj_diff_min) {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if (Gmax+Gmax2 < eps)
    return 1;

  out_i = Gmax_idx;
  out_j = Gmin_idx;
  return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2) {
  if (is_upper_bound(i)) {
    if (y[i] == +1) {
      return(-G[i] > Gmax1);
    } else {
      return(-G[i] > Gmax2);
    }
  } else {
    if (is_lower_bound(i)) {
      if (y[i] == +1) {
        return(G[i] > Gmax2);
      } else {
        return(G[i] > Gmax1);
      }
    } else {
      return(false);
    }
  }
}

void Solver::do_shrinking() {
  int i;
  double Gmax1 = -INF;    // max { -y_i * grad(f)_i | i in I_up(\alpha) }
  double Gmax2 = -INF;    // max { y_i * grad(f)_i | i in I_low(\alpha) }

  // find maximal violating pair first
  for (i = 0; i < active_size; ++i) {
    if (y[i] == +1) {
      if (!is_upper_bound(i)) {
        if (-G[i] >= Gmax1)
          Gmax1 = -G[i];
      }
      if (!is_lower_bound(i)) {
        if (G[i] >= Gmax2)
          Gmax2 = G[i];
      }
    } else {
      if (!is_upper_bound(i)) {
        if (-G[i] >= Gmax2)
          Gmax2 = -G[i];
      }
      if (!is_lower_bound(i)) {
        if (G[i] >= Gmax1)
          Gmax1 = G[i];
      }
    }
  }

  if (unshrink == false && Gmax1 + Gmax2 <= eps*10) {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
    info("*");
  }

  for (i = 0; i < active_size; ++i)
    if (be_shrunk(i, Gmax1, Gmax2)) {
      active_size--;
      while (active_size > i) {
        if (!be_shrunk(active_size, Gmax1, Gmax2)) {
          swap_index(i, active_size);
          break;
        }
        active_size--;
      }
    }
}

double Solver::calculate_rho() {
  double r;
  int nr_free = 0;
  double ub = INF, lb = -INF, sum_free = 0;
  for (int i = 0; i < active_size; ++i) {
    double yG = y[i]*G[i];

    if (is_upper_bound(i)) {
      if (y[i] == -1) {
        ub = std::min(ub,yG);
      } else {
        lb = std::max(lb,yG);
      }
    } else {
      if (is_lower_bound(i)) {
        if (y[i] == +1) {
          ub = std::min(ub,yG);
        } else {
          lb = std::max(lb,yG);
        }
      } else {
        ++nr_free;
        sum_free += yG;
      }
    }
  }

  if (nr_free > 0) {
    r = sum_free/nr_free;
  } else {
    r = (ub+lb)/2;
  }

  return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver {
 public:
  Solver_NU() {}
  void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
       double *alpha, double Cp, double Cn, double eps,
       SolutionInfo* si, int shrinking) {
    this->si = si;
    Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
  }

 private:
  SolutionInfo *si;
  int select_working_set(int &i, int &j);
  double calculate_rho();
  bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
  void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j) {
  // return i,j such that y_i = y_j and
  // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  // j: minimizes the decrease of obj value
  //    (if quadratic coefficeint <= 0, replace it with tau)
  //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

  double Gmaxp = -INF;
  double Gmaxp2 = -INF;
  int Gmaxp_idx = -1;

  double Gmaxn = -INF;
  double Gmaxn2 = -INF;
  int Gmaxn_idx = -1;

  int Gmin_idx = -1;
  double obj_diff_min = INF;

  for (int t = 0; t < active_size; ++t)
    if (y[t] == +1) {
      if (!is_upper_bound(t))
        if (-G[t] >= Gmaxp) {
          Gmaxp = -G[t];
          Gmaxp_idx = t;
        }
    } else {
      if (!is_lower_bound(t))
        if (G[t] >= Gmaxn) {
          Gmaxn = G[t];
          Gmaxn_idx = t;
        }
    }

  int ip = Gmaxp_idx;
  int in = Gmaxn_idx;
  const Qfloat *Q_ip = NULL;
  const Qfloat *Q_in = NULL;
  if (ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
    Q_ip = Q->get_Q(ip,active_size);
  if (in != -1)
    Q_in = Q->get_Q(in,active_size);

  for (int j = 0; j < active_size; ++j) {
    if (y[j] == +1) {
      if (!is_lower_bound(j)) {
        double grad_diff=Gmaxp+G[j];
        if (G[j] >= Gmaxp2)
          Gmaxp2 = G[j];
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff)/TAU;
          }

          if (obj_diff <= obj_diff_min) {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    } else {
      if (!is_upper_bound(j)) {
        double grad_diff=Gmaxn-G[j];
        if (-G[j] >= Gmaxn2)
          Gmaxn2 = -G[j];
        if (grad_diff > 0) {
          double obj_diff;
          double quad_coef = QD[in]+QD[j]-2*Q_in[j];
          if (quad_coef > 0) {
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          } else {
            obj_diff = -(grad_diff*grad_diff)/TAU;
          }

          if (obj_diff <= obj_diff_min) {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if (std::max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
    return 1;

  if (y[Gmin_idx] == +1) {
    out_i = Gmaxp_idx;
  } else {
    out_i = Gmaxn_idx;
  }
  out_j = Gmin_idx;

  return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4) {
  if (is_upper_bound(i)) {
    if (y[i] == +1) {
      return(-G[i] > Gmax1);
    } else {
      return(-G[i] > Gmax4);
    }
  } else {
    if (is_lower_bound(i)) {
      if (y[i] == +1) {
        return(G[i] > Gmax2);
      } else {
        return(G[i] > Gmax3);
      }
    } else {
      return(false);
    }
  }
}

void Solver_NU::do_shrinking() {
  double Gmax1 = -INF;  // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
  double Gmax2 = -INF;  // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
  double Gmax3 = -INF;  // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
  double Gmax4 = -INF;  // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

  // find maximal violating pair first
  int i;
  for (i = 0; i < active_size; ++i) {
    if (!is_upper_bound(i)) {
      if (y[i] == +1) {
        if (-G[i] > Gmax1)
          Gmax1 = -G[i];
      } else {
        if (-G[i] > Gmax4)
          Gmax4 = -G[i];
      }
    }
    if (!is_lower_bound(i)) {
      if (y[i] == +1) {
        if(G[i] > Gmax2)
          Gmax2 = G[i];
      } else {
        if (G[i] > Gmax3)
          Gmax3 = G[i];
      }
    }
  }

  if (unshrink == false && std::max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
  }

  for (i = 0; i < active_size; ++i)
    if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4)) {
      active_size--;
      while (active_size > i) {
        if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4)) {
          swap_index(i,active_size);
          break;
        }
        active_size--;
      }
    }
}

double Solver_NU::calculate_rho() {
  int nr_free1 = 0,nr_free2 = 0;
  double ub1 = INF, ub2 = INF;
  double lb1 = -INF, lb2 = -INF;
  double sum_free1 = 0, sum_free2 = 0;

  for (int i = 0; i < active_size; ++i) {
    if (y[i] == +1) {
      if (is_upper_bound(i)) {
        lb1 = std::max(lb1,G[i]);
      } else {
        if (is_lower_bound(i)) {
          ub1 = std::min(ub1,G[i]);
        } else {
          ++nr_free1;
          sum_free1 += G[i];
        }
      }
    } else {
      if (is_upper_bound(i)) {
        lb2 = std::max(lb2,G[i]);
      } else {
        if (is_lower_bound(i)) {
          ub2 = std::min(ub2,G[i]);
        } else {
          ++nr_free2;
          sum_free2 += G[i];
        }
      }
    }
  }

  double r1,r2;
  if (nr_free1 > 0) {
    r1 = sum_free1/nr_free1;
  } else {
    r1 = (ub1+lb1)/2;
  }

  if (nr_free2 > 0) {
    r2 = sum_free2/nr_free2;
  } else {
    r2 = (ub2+lb2)/2;
  }

  si->r = (r1+r2)/2;
  return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q : public Kernel {
 public:
  SVC_Q(const Problem &prob, const SVMParameter &param, const schar *y_) : Kernel(prob.num_ex, prob.x, param) {
    clone(y, y_, prob.num_ex);
    cache = new Cache(prob.num_ex, static_cast<long int>(param.cache_size*(1<<20)));
    QD = new double[prob.num_ex];
    for (int i = 0; i < prob.num_ex; ++i)
      QD[i] = (this->*kernel_function)(i, i);
  }

  Qfloat *get_Q(int i, int len) const {
    Qfloat *data;
    int start = cache->get_data(i, &data, len);
    if (start < len) {
      for (int j = start; j < len; ++j)
        data[j] = static_cast<Qfloat>(y[i]*y[j]*(this->*kernel_function)(i, j));
    }
    return data;
  }

  double *get_QD() const {
    return QD;
  }

  void swap_index(int i, int j) const {
    cache->swap_index(i, j);
    Kernel::swap_index(i, j);
    std::swap(y[i], y[j]);
    std::swap(QD[i], QD[j]);
  }

  ~SVC_Q() {
    delete[] y;
    delete cache;
    delete[] QD;
  }

 private:
  schar *y;
  Cache *cache;
  double *QD;
};

//
// construct and solve various formulations
//
static void SolveCSVC(const Problem *prob, const SVMParameter *param, double *alpha, Solver::SolutionInfo *si, double Cp, double Cn) {
  int num_ex = prob->num_ex;
  double *minus_ones = new double[num_ex];
  schar *y = new schar[num_ex];

  for (int i = 0; i < num_ex; ++i) {
    alpha[i] = 0;
    minus_ones[i] = -1;
    if (prob->y[i] > 0) {
      y[i] = +1;
    } else {
      y[i] = -1;
    }
  }

  Solver s;
  s.Solve(num_ex, SVC_Q(*prob, *param,y), minus_ones, y, alpha, Cp, Cn, param->eps, si, param->shrinking);

  double sum_alpha=0;
  for (int i = 0; i < num_ex; ++i)
    sum_alpha += alpha[i];

  if (Cp == Cn)
    info("nu = %f\n", sum_alpha/(Cp*prob->num_ex));

  for (int i = 0; i < num_ex; ++i)
    alpha[i] *= y[i];

  delete[] minus_ones;
  delete[] y;
}

static void SolveNuSVC(const Problem *prob, const SVMParameter *param, double *alpha, Solver::SolutionInfo *si) {
  int num_ex = prob->num_ex;
  double nu = param->nu;

  schar *y = new schar[num_ex];

  for (int i = 0; i < num_ex; ++i)
    if (prob->y[i] > 0) {
      y[i] = +1;
    } else {
      y[i] = -1;
    }

  double sum_pos = nu*num_ex/2;
  double sum_neg = nu*num_ex/2;

  for (int i = 0; i < num_ex; ++i)
    if (y[i] == +1) {
      alpha[i] = std::min(1.0, sum_pos);
      sum_pos -= alpha[i];
    } else {
      alpha[i] = std::min(1.0, sum_neg);
      sum_neg -= alpha[i];
    }

  double *zeros = new double[num_ex];

  for (int i = 0; i < num_ex; ++i)
    zeros[i] = 0;

  Solver_NU s;
  s.Solve(num_ex, SVC_Q(*prob, *param,y), zeros, y, alpha, 1.0, 1.0, param->eps, si, param->shrinking);
  double r = si->r;

  info("C = %f\n", 1/r);

  for (int i = 0; i < num_ex; ++i)
    alpha[i] *= y[i]/r;

  si->rho /= r;
  si->obj /= (r*r);
  si->upper_bound_p = 1/r;
  si->upper_bound_n = 1/r;

  delete[] y;
  delete[] zeros;
}

//
// DecisionFunction
//
struct DecisionFunction {
  double *alpha;
  double rho;
};

static DecisionFunction TrainSingleSVM(const Problem *prob, const SVMParameter *param, double Cp, double Cn) {
  double *alpha = new double[prob->num_ex];
  Solver::SolutionInfo si;
  switch (param->svm_type) {
    case C_SVC: {
      SolveCSVC(prob, param, alpha, &si, Cp, Cn);
      break;
    }
    case NU_SVC: {
      SolveNuSVC(prob, param, alpha, &si);
      break;
    }
    default: {
      // assert{false};
      break;
    }
  }

  info("obj = %f, rho = %f\n", si.obj, si.rho);

  // output SVs
  int nSV = 0;
  int nBSV = 0;
  for (int i = 0; i < prob->num_ex; ++i) {
    if (fabs(alpha[i]) > 0) {
      ++nSV;
      if (prob->y[i] > 0) {
        if (fabs(alpha[i]) >= si.upper_bound_p)
          ++nBSV;
      } else {
        if (fabs(alpha[i]) >= si.upper_bound_n)
          ++nBSV;
      }
    }
  }

  info("nSV = %d, nBSV = %d\n", nSV, nBSV);

  DecisionFunction f;
  f.alpha = alpha;
  f.rho = si.rho;
  return f;
}

//
// Interface functions
//
SVMModel *TrainSVM(const Problem *prob, const SVMParameter *param) {
  SVMModel *model = new SVMModel;
  model->param = *param;
  model->free_sv = 0;  // XXX

  // classification
  int num_ex = prob->num_ex;
  int num_classes;
  int *labels = NULL;
  int *start = NULL;
  int *count = NULL;
  int *perm = new int[num_ex];

  // group training data of the same class
  GroupClasses(prob,&num_classes,&labels,&start,&count,perm);
  if (num_classes == 1)
    info("WARNING: training data in only one class. See README for details.\n");

  Node **x = new Node*[num_ex];
  for (int i = 0; i < num_ex; ++i)
    x[i] = prob->x[perm[i]];

  // calculate weighted C
  double *weighted_C = new double[num_classes];
  for (int i = 0; i < num_classes; ++i)
    weighted_C[i] = param->C;
  for (int i = 0; i < param->num_weights; ++i) {
    int j;
    for (j = 0; j < num_classes; ++j)
      if (param->weight_labels[i] == labels[j]) break;
    if (j == num_classes) {
      fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_labels[i]);
    } else {
      weighted_C[j] *= param->weights[i];
    }
  }

  // train k*(k-1)/2 models
  bool *nonzero = new bool[num_ex];
  for (int i = 0; i < num_ex; ++i)
    nonzero[i] = false;
  DecisionFunction *f = new DecisionFunction[num_classes*(num_classes-1)/2];

  int p = 0;
  for (int i = 0; i < num_classes; ++i)
    for (int j = i+1; j < num_classes; ++j) {
      Problem sub_prob;
      int si = start[i], sj = start[j];
      int ci = count[i], cj = count[j];
      sub_prob.num_ex = ci+cj;
      sub_prob.x = new Node *[sub_prob.num_ex];
      sub_prob.y = new double[sub_prob.num_ex];
      for (int k = 0; k < ci; ++k) {
        sub_prob.x[k] = x[si+k];
        sub_prob.y[k] = +1;
      }
      for (int k = 0; k < cj; ++k) {
        sub_prob.x[ci+k] = x[sj+k];
        sub_prob.y[ci+k] = -1;
      }

      f[p] = TrainSingleSVM(&sub_prob, param,weighted_C[i], weighted_C[j]);
      for (int k = 0; k < ci; ++k)
        if (!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
          nonzero[si+k] = true;
      for (int k = 0; k < cj; ++k)
        if (!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
          nonzero[sj+k] = true;
      delete[] sub_prob.x;
      delete[] sub_prob.y;
      ++p;
    }

  // build output
  model->num_classes = num_classes;
  model->num_ex = num_ex;

  model->labels = new int[num_classes];
  for (int i = 0; i < num_classes; ++i)
    model->labels[i] = labels[i];

  model->rho = new double[num_classes*(num_classes-1)/2];
  for (int i = 0; i < num_classes*(num_classes-1)/2; ++i)
    model->rho[i] = f[i].rho;

  int total_sv = 0;
  int *nz_count = new int[num_classes];
  model->num_svs = new int[num_classes];
  for (int i = 0; i < num_classes; ++i) {
    int num_svs = 0;
    for (int j = 0; j < count[i]; ++j)
      if (nonzero[start[i]+j]) {
        ++num_svs;
        ++total_sv;
      }
    model->num_svs[i] = num_svs;
    nz_count[i] = num_svs;
  }

  info("Total nSV = %d\n",total_sv);

  model->total_sv = total_sv;
  model->svs = new Node*[total_sv];
  model->sv_indices = new int[total_sv];
  p = 0;
  for (int i = 0; i < num_ex; ++i)
    if (nonzero[i]) {
      model->svs[p] = x[i];
      model->sv_indices[p] = perm[i] + 1;
      ++p;
    }

  int *nz_start = new int[num_classes];
  nz_start[0] = 0;
  for (int i = 1; i < num_classes; ++i)
    nz_start[i] = nz_start[i-1]+nz_count[i-1];

  model->sv_coef = new double*[num_classes-1];
  for (int i = 0; i < num_classes-1; ++i)
    model->sv_coef[i] = new double[total_sv];

  p = 0;
  for (int i = 0; i < num_classes; ++i)
    for (int j = i+1; j < num_classes; ++j) {
      // classifier (i,j): coefficients with
      // i are in sv_coef[j-1][nz_start[i]...],
      // j are in sv_coef[i][nz_start[j]...]
      int si = start[i];
      int sj = start[j];
      int ci = count[i];
      int cj = count[j];

      int q = nz_start[i];
      for (int k = 0; k < ci; ++k)
        if (nonzero[si+k])
          model->sv_coef[j-1][q++] = f[p].alpha[k];
      q = nz_start[j];
      for (int k = 0; k < cj; ++k)
        if (nonzero[sj+k])
          model->sv_coef[i][q++] = f[p].alpha[ci+k];
      ++p;
    }

  delete[] labels;
  delete[] count;
  delete[] perm;
  delete[] start;
  delete[] x;
  delete[] weighted_C;
  delete[] nonzero;
  for (int i = 0; i < num_classes*(num_classes-1)/2; ++i)
    delete[] f[i].alpha;
  delete[] f;
  delete[] nz_count;
  delete[] nz_start;

  return model;
}

double PredictSVMValues(const SVMModel *model, const Node *x, double *decision_values) {
  int num_classes = model->num_classes;
  int total_sv = model->total_sv;

  double *kvalue = new double[total_sv];
  for (int i = 0; i < total_sv; ++i)
    kvalue[i] = Kernel::k_function(x, model->svs[i], model->param);

  int *start = new int[num_classes];
  start[0] = 0;
  for (int i = 1; i < num_classes; ++i)
    start[i] = start[i-1] + model->num_svs[i-1];

  int *vote = new int[num_classes];
  for (int i = 0; i < num_classes; ++i)
    vote[i] = 0;

  int p = 0;
  for (int i = 0; i < num_classes; ++i)
    for (int j = i+1; j < num_classes; ++j) {
      double sum = 0;
      int si = start[i];
      int sj = start[j];
      int ci = model->num_svs[i];
      int cj = model->num_svs[j];

      double *coef1 = model->sv_coef[j-1];
      double *coef2 = model->sv_coef[i];
      for (int k = 0; k < ci; ++k)
        sum += coef1[si+k] * kvalue[si+k];
      for (int k = 0; k < cj; ++k)
        sum += coef2[sj+k] * kvalue[sj+k];
      sum -= model->rho[p];
      decision_values[p] = sum;

      if (decision_values[p] > 0) {
        ++vote[i];
      } else {
        ++vote[j];
      }
      ++p;
    }

  int vote_max_idx = 0;
  for (int i = 1; i < num_classes; ++i)
    if (vote[i] > vote[vote_max_idx])
      vote_max_idx = i;

  delete[] kvalue;
  delete[] start;
  delete[] vote;

  return model->labels[vote_max_idx];
}

double PredictSVM(const SVMModel *model, const Node *x) {
  int num_classes = model->num_classes;
  double *decision_values = new double[num_classes*(num_classes-1)/2];
  double pred_result = PredictSVMValues(model, x, decision_values);
  delete[] decision_values;
  return pred_result;
}

double PredictDecisionValues(const struct SVMModel *model, const struct Node *x, double **decision_values) {
  int num_classes = model->num_classes;
  *decision_values = new double[num_classes*(num_classes-1)/2];
  double pred_result = PredictSVMValues(model, x, *decision_values);
  return pred_result;
}

static const char *kSVMTypeTable[] = { "c_svc", "nu_svc", NULL };

static const char *kKernelTypeTable[] = { "linear", "polynomial", "rbf", "sigmoid", "precomputed", NULL };

int SaveSVMModel(std::ofstream &model_file, const struct SVMModel *model) {
  const SVMParameter &param = model->param;

  model_file << "svm_model\n";
  model_file << "svm_type " << kSVMTypeTable[param.svm_type] << '\n';
  model_file << "kernel_type " << kKernelTypeTable[param.kernel_type] << '\n';

  if (param.kernel_type == POLY) {
    model_file << "degree " << param.degree << '\n';
  }
  if (param.kernel_type == POLY ||
      param.kernel_type == RBF  ||
      param.kernel_type == SIGMOID) {
    model_file << "gamma " << param.gamma << '\n';
  }
  if (param.kernel_type == POLY ||
      param.kernel_type == SIGMOID) {
    model_file << "coef0 " << param.coef0 << '\n';
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

  if (model->rho) {
    model_file << "rho";
    for (int i = 0; i < num_classes*(num_classes-1)/2; ++i)
      model_file << ' ' << model->rho[i];
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
  const double *const *sv_coef = model->sv_coef;
  const Node *const *svs = model->svs;

  for (int i = 0; i < total_sv; ++i) {
    for (int j = 0; j < num_classes-1; ++j)
      model_file << std::setprecision(16) << (sv_coef[j][i]+0.0) << ' ';  // add "+0.0" to avoid negative zero in output

    const Node *p = svs[i];

    if (param.kernel_type == PRECOMPUTED) {
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

SVMModel *LoadSVMModel(std::ifstream &model_file) {
  SVMModel *model = new SVMModel;
  SVMParameter &param = model->param;
  model->rho = NULL;
  model->sv_indices = NULL;
  model->labels = NULL;
  model->num_svs = NULL;

  char cmd[80];
  while (1) {
    model_file >> cmd;

    if (std::strcmp(cmd, "svm_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kSVMTypeTable[i]; ++i) {
        if (std::strcmp(kSVMTypeTable[i], cmd) == 0) {
          param.svm_type = i;
          break;
        }
      }
      if (kSVMTypeTable[i] == NULL) {
        std::cerr << "Unknown SVM type.\n" << std::endl;
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "kernel_type") == 0) {
      model_file >> cmd;
      int i;
      for (i = 0; kKernelTypeTable[i]; ++i) {
        if (std::strcmp(kKernelTypeTable[i], cmd) == 0) {
          param.kernel_type = i;
          break;
        }
      }
      if (kKernelTypeTable[i] == NULL) {
        std::cerr << "Unknown kernel function.\n" << std::endl;
        return NULL;
      }
    } else
    if (std::strcmp(cmd, "degree") == 0) {
      model_file >> param.degree;
    } else
    if (std::strcmp(cmd, "gamma") == 0) {
      model_file >> param.gamma;
    } else
    if (std::strcmp(cmd, "coef0") == 0) {
      model_file >> param.coef0;
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
    if (std::strcmp(cmd, "rho") == 0) {
      int n = model->num_classes*(model->num_classes-1)/2;
      model->rho = new double[n];
      for (int i = 0; i < n; ++i) {
        model_file >> model->rho[i];
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
      std::size_t m = static_cast<unsigned long>(model->num_classes)-1;
      int total_sv = model->total_sv;
      std::string line;

      if (model_file.peek() == '\n')
        model_file.get();

      model->sv_coef = new double*[m];
      for (int i = 0; i < m; ++i) {
        model->sv_coef[i] = new double[total_sv];
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

        for (std::size_t j = 0; j < m; ++j) {
          try
          {
            std::size_t end;
            model->sv_coef[j][i] = std::stod(tokens[j], &end);
            if (end != tokens[j].length()) {
              throw std::invalid_argument("incomplete convention");
            }
          }
          catch(std::exception& e)
          {
            std::cerr << "Error: " << e.what() << " in SV " << (i+1) << std::endl;
            delete[] model->svs;
            for (int j = 0; j < m; ++j) {
              delete[] model->sv_coef[j];
            }
            delete[] model->sv_coef;
            std::vector<std::string>(tokens).swap(tokens);
            exit(EXIT_FAILURE);
          }  // TODO try not to use exception
        }

        std::size_t elements = tokens.size() - m + 1;
        model->svs[i] = new Node[elements];
        prev = 0;
        for (std::size_t j = 0; j < elements-1; ++j) {
          pos = tokens[j+m].find_first_of(':');
          try
          {
            std::size_t end;

            model->svs[i][j].index = std::stoi(tokens[j+m].substr(prev, pos-prev), &end);
            if (end != (tokens[j+m].substr(prev, pos-prev)).length()) {
              throw std::invalid_argument("incomplete convention");
            }
            model->svs[i][j].value = std::stod(tokens[j+m].substr(pos+1), &end);
            if (end != (tokens[j+m].substr(pos+1)).length()) {
              throw std::invalid_argument("incomplete convention");
            }
          }
          catch(std::exception& e)
          {
            std::cerr << "Error: " << e.what() << " in line " << (i+1) << std::endl;
            for (int k = 0; k < m; ++k) {
              delete[] model->sv_coef[k];
            }
            delete[] model->sv_coef;
            for (int k = 0; k < i+1; ++k) {
              delete[] model->svs[k];
            }
            delete[] model->svs;
            std::vector<std::string>(tokens).swap(tokens);
            exit(EXIT_FAILURE);
          }
        }
        model->svs[i][elements-1].index = -1;
        model->svs[i][elements-1].value = 0;
      }
      break;
    } else {
      std::cerr << "Unknown text in knn_model file: " << cmd << std::endl;
      FreeSVMModel(&model);
      return NULL;
    }
  }
  model->free_sv = 1;
  return model;
}

void FreeSVMModelContent(SVMModel *model) {
  if (model->free_sv && model->total_sv > 0 && model->svs != NULL) {
    delete[] model->svs;
    model->svs = NULL;
  }

  if (model->sv_coef) {
    for (int i = 0; i < model->num_classes-1; ++i)
      delete[] model->sv_coef[i];
  }

  if (model->svs) {
    delete[] model->svs;
    model->svs = NULL;
  }

  if (model->sv_coef) {
    delete[] model->sv_coef;
    model->sv_coef = NULL;
  }

  if (model->rho) {
    delete[] model->rho;
    model->rho = NULL;
  }

  if (model->labels) {
    delete[] model->labels;
    model->labels= NULL;
  }

  if (model->sv_indices) {
    delete[] model->sv_indices;
    model->sv_indices = NULL;
  }

  if (model->num_svs) {
    delete[] model->num_svs;
    model->num_svs = NULL;
  }
}

void FreeSVMModel(SVMModel** model)
{
  if (model != NULL && *model != NULL) {
    FreeSVMModelContent(*model);
    delete *model;
    *model = NULL;
  }

  return;
}

void FreeSVMParam(SVMParameter* param) {
  if (param->weight_labels) {
    delete[] param->weight_labels;
    param->weight_labels = NULL;
  }
  if (param->weights) {
    delete[] param->weights;
    param->weights = NULL;
  }
  delete param;
  param = NULL;

  return;
}

const char *CheckSVMParameter(const SVMParameter *param) {
  int svm_type = param->svm_type;
  if (svm_type != C_SVC &&
      svm_type != NU_SVC)
    return "unknown svm type";

  int kernel_type = param->kernel_type;
  if (kernel_type != LINEAR &&
      kernel_type != POLY &&
      kernel_type != RBF &&
      kernel_type != SIGMOID &&
      kernel_type != PRECOMPUTED)
    return "unknown kernel type";

  if (param->gamma < 0)
    return "gamma < 0";

  if (param->degree < 0)
    return "degree of polynomial kernel < 0";

  if (param->cache_size <= 0)
    return "cache_size <= 0";

  if (param->eps <= 0)
    return "eps <= 0";

  if (svm_type == C_SVC)
    if (param->C <= 0)
      return "C <= 0";

  if (svm_type == NU_SVC)
    if (param->nu <= 0 || param->nu > 1)
      return "nu <= 0 or nu > 1";

  if (param->shrinking != 0 &&
      param->shrinking != 1)
    return "shrinking != 0 and shrinking != 1";

  return NULL;
}

void InitSVMParam(struct SVMParameter *param) {
  param->svm_type = C_SVC;
  param->kernel_type = RBF;
  param->degree = 3;
  param->gamma = 0.1;  // 1/num_features
  param->coef0 = 0;
  param->nu = 0.5;
  param->cache_size = 100;
  param->C = 1;
  param->eps = 1e-3;
  param->shrinking = 1;
  param->num_weights = 0;
  param->weight_labels = NULL;
  param->weights = NULL;
  SetPrintCout();

  return;
}

void SetPrintNull() {
  PrintString = &PrintNull;
}

void SetPrintCout() {
  PrintString = &PrintCout;
}