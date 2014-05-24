#include "svm.h"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>

typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
static inline double powi(double base, int times)
{
  double tmp = base, ret = 1.0;

  for(int t=times; t>0; t/=2)
  {
    if(t%2==1) ret*=tmp;
    tmp = tmp * tmp;
  }
  return ret;
}
#define Malloc(type,n) (type *)malloc((unsigned long)(n)*sizeof(type))

static void print_string_stdout(const char *s)
{
  fputs(s,stdout);
  fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
  char buf[BUFSIZ];
  va_list ap;
  va_start(ap,fmt);
  vsprintf(buf,fmt,ap);
  va_end(ap);
  (*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
  Cache(int l,long int size);
  ~Cache();

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  int get_data(const int index, Qfloat **data, int len);
  void swap_index(int i, int j);
private:
  int l;
  long int size;
  struct head_t
  {
    head_t *prev, *next;  // a circular list
    Qfloat *data;
    int len;    // data[0,len) is cached in this entry
  };

  head_t *head;
  head_t lru_head;
  void lru_delete(head_t *h);
  void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_)
{
  head = (head_t *)calloc((size_t)l,sizeof(head_t));  // initialized to 0
  size /= sizeof(Qfloat);
  size -= (unsigned long)l * sizeof(head_t) / sizeof(Qfloat);
  size = max(size, 2 * (long int) l);  // cache must be large enough for two columns
  lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache()
{
  for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
    free(h->data);
  free(head);
}

void Cache::lru_delete(head_t *h)
{
  // delete from current location
  h->prev->next = h->next;
  h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h)
{
  // insert to last position
  h->next = &lru_head;
  h->prev = lru_head.prev;
  h->prev->next = h;
  h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len)
{
  head_t *h = &head[index];
  if(h->len) lru_delete(h);
  int more = len - h->len;

  if(more > 0)
  {
    // free old space
    while(size < more)
    {
      head_t *old = lru_head.next;
      lru_delete(old);
      free(old->data);
      size += old->len;
      old->data = 0;
      old->len = 0;
    }

    // allocate new space
    h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*(unsigned long)len);
    size -= more;
    swap(h->len,len);
  }

  lru_insert(h);
  *data = h->data;
  return len;
}

void Cache::swap_index(int i, int j)
{
  if(i==j) return;

  if(head[i].len) lru_delete(&head[i]);
  if(head[j].len) lru_delete(&head[j]);
  swap(head[i].data,head[j].data);
  swap(head[i].len,head[j].len);
  if(head[i].len) lru_insert(&head[i]);
  if(head[j].len) lru_insert(&head[j]);

  if(i>j) swap(i,j);
  for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
  {
    if(h->len > i)
    {
      if(h->len > j)
        swap(h->data[i],h->data[j]);
      else
      {
        // give up
        lru_delete(h);
        free(h->data);
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

class Kernel: public QMatrix {
public:
  Kernel(int l, Node * const * x, const SVMParameter& param);
  virtual ~Kernel();

  static double k_function(const Node *x, const Node *y,
         const SVMParameter& param);
  virtual Qfloat *get_Q(int column, int len) const = 0;
  virtual double *get_QD() const = 0;
  virtual void swap_index(int i, int j) const  // no so const...
  {
    swap(x[i],x[j]);
    if(x_square) swap(x_square[i],x_square[j]);
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
  double kernel_linear(int i, int j) const
  {
    return dot(x[i],x[j]);
  }
  double kernel_poly(int i, int j) const
  {
    return powi(gamma*dot(x[i],x[j])+coef0,degree);
  }
  double kernel_rbf(int i, int j) const
  {
    return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
  }
  double kernel_sigmoid(int i, int j) const
  {
    return tanh(gamma*dot(x[i],x[j])+coef0);
  }
  double kernel_precomputed(int i, int j) const
  {
    return x[i][(int)(x[j][0].value)].value;
  }
};

Kernel::Kernel(int l, Node * const * x_, const SVMParameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
  switch(kernel_type)
  {
    case LINEAR:
      kernel_function = &Kernel::kernel_linear;
      break;
    case POLY:
      kernel_function = &Kernel::kernel_poly;
      break;
    case RBF:
      kernel_function = &Kernel::kernel_rbf;
      break;
    case SIGMOID:
      kernel_function = &Kernel::kernel_sigmoid;
      break;
    case PRECOMPUTED:
      kernel_function = &Kernel::kernel_precomputed;
      break;
  }

  clone(x,x_,l);

  if(kernel_type == RBF)
  {
    x_square = new double[l];
    for(int i=0;i<l;i++)
      x_square[i] = dot(x[i],x[i]);
  }
  else
    x_square = 0;
}

Kernel::~Kernel()
{
  delete[] x;
  delete[] x_square;
}

double Kernel::dot(const Node *px, const Node *py)
{
  double sum = 0;
  while(px->index != -1 && py->index != -1)
  {
    if(px->index == py->index)
    {
      sum += px->value * py->value;
      ++px;
      ++py;
    }
    else
    {
      if(px->index > py->index)
        ++py;
      else
        ++px;
    }
  }
  return sum;
}

double Kernel::k_function(const Node *x, const Node *y,
        const SVMParameter& param)
{
  switch(param.kernel_type)
  {
    case LINEAR:
      return dot(x,y);
    case POLY:
      return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
    case RBF:
    {
      double sum = 0;
      while(x->index != -1 && y->index !=-1)
      {
        if(x->index == y->index)
        {
          double d = x->value - y->value;
          sum += d*d;
          ++x;
          ++y;
        }
        else
        {
          if(x->index > y->index)
          {
            sum += y->value * y->value;
            ++y;
          }
          else
          {
            sum += x->value * x->value;
            ++x;
          }
        }
      }

      while(x->index != -1)
      {
        sum += x->value * x->value;
        ++x;
      }

      while(y->index != -1)
      {
        sum += y->value * y->value;
        ++y;
      }

      return exp(-param.gamma*sum);
    }
    case SIGMOID:
      return tanh(param.gamma*dot(x,y)+param.coef0);
    case PRECOMPUTED:  //x: test (validation), y: SV
      return x[(int)(y->value)].value;
    default:
      return 0;  // Unreachable
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

  void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
       double *alpha_, double Cp, double Cn, double eps,
       SolutionInfo* si, int shrinking);
protected:
  int active_size;
  schar *y;
  double *G;    // gradient of objective function
  enum { LOWER_BOUND, UPPER_BOUND, FREE };
  char *alpha_status;  // LOWER_BOUND, UPPER_BOUND, FREE
  double *alpha;
  const QMatrix *Q;
  const double *QD;
  double eps;
  double Cp,Cn;
  double *p;
  int *active_set;
  double *G_bar;    // gradient, if we treat free variables as 0
  int l;
  bool unshrink;  // XXX

  double get_C(int i)
  {
    return (y[i] > 0)? Cp : Cn;
  }
  void update_alpha_status(int i)
  {
    if(alpha[i] >= get_C(i))
      alpha_status[i] = UPPER_BOUND;
    else if(alpha[i] <= 0)
      alpha_status[i] = LOWER_BOUND;
    else alpha_status[i] = FREE;
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

void Solver::swap_index(int i, int j)
{
  Q->swap_index(i,j);
  swap(y[i],y[j]);
  swap(G[i],G[j]);
  swap(alpha_status[i],alpha_status[j]);
  swap(alpha[i],alpha[j]);
  swap(p[i],p[j]);
  swap(active_set[i],active_set[j]);
  swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
  // reconstruct inactive elements of G from G_bar and free variables

  if(active_size == l) return;

  int i,j;
  int nr_free = 0;

  for(j=active_size;j<l;j++)
    G[j] = G_bar[j] + p[j];

  for(j=0;j<active_size;j++)
    if(is_free(j))
      nr_free++;

  if(2*nr_free < active_size)
    info("\nWARNING: using -h 0 may be faster\n");

  if (nr_free*l > 2*active_size*(l-active_size))
  {
    for(i=active_size;i<l;i++)
    {
      const Qfloat *Q_i = Q->get_Q(i,active_size);
      for(j=0;j<active_size;j++)
        if(is_free(j))
          G[i] += alpha[j] * Q_i[j];
    }
  }
  else
  {
    for(i=0;i<active_size;i++)
      if(is_free(i))
      {
        const Qfloat *Q_i = Q->get_Q(i,l);
        double alpha_i = alpha[i];
        for(j=active_size;j<l;j++)
          G[j] += alpha_i * Q_i[j];
      }
  }
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
       double *alpha_, double Cp, double Cn, double eps,
       SolutionInfo* si, int shrinking)
{
  this->l = l;
  this->Q = &Q;
  QD=Q.get_QD();
  clone(p, p_,l);
  clone(y, y_,l);
  clone(alpha,alpha_,l);
  this->Cp = Cp;
  this->Cn = Cn;
  this->eps = eps;
  unshrink = false;

  // initialize alpha_status
  {
    alpha_status = new char[l];
    for(int i=0;i<l;i++)
      update_alpha_status(i);
  }

  // initialize active set (for shrinking)
  {
    active_set = new int[l];
    for(int i=0;i<l;i++)
      active_set[i] = i;
    active_size = l;
  }

  // initialize gradient
  {
    G = new double[l];
    G_bar = new double[l];
    int i;
    for(i=0;i<l;i++)
    {
      G[i] = p[i];
      G_bar[i] = 0;
    }
    for(i=0;i<l;i++)
      if(!is_lower_bound(i))
      {
        const Qfloat *Q_i = Q.get_Q(i,l);
        double alpha_i = alpha[i];
        int j;
        for(j=0;j<l;j++)
          G[j] += alpha_i*Q_i[j];
        if(is_upper_bound(i))
          for(j=0;j<l;j++)
            G_bar[j] += get_C(i) * Q_i[j];
      }
  }

  // optimization step

  int iter = 0;
  int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
  int counter = min(l,1000)+1;

  while(iter < max_iter)
  {
    // show progress and do shrinking

    if(--counter == 0)
    {
      counter = min(l,1000);
      if(shrinking) do_shrinking();
      info(".");
    }

    int i,j;
    if(select_working_set(i,j)!=0)
    {
      // reconstruct the whole gradient
      reconstruct_gradient();
      // reset active set size and check
      active_size = l;
      info("*");
      if(select_working_set(i,j)!=0)
        break;
      else
        counter = 1;  // do shrinking next iteration
    }

    ++iter;

    // update alpha[i] and alpha[j], handle bounds carefully

    const Qfloat *Q_i = Q.get_Q(i,active_size);
    const Qfloat *Q_j = Q.get_Q(j,active_size);

    double C_i = get_C(i);
    double C_j = get_C(j);

    double old_alpha_i = alpha[i];
    double old_alpha_j = alpha[j];

    if(y[i]!=y[j])
    {
      double quad_coef = QD[i]+QD[j]+2*Q_i[j];
      if (quad_coef <= 0)
        quad_coef = TAU;
      double delta = (-G[i]-G[j])/quad_coef;
      double diff = alpha[i] - alpha[j];
      alpha[i] += delta;
      alpha[j] += delta;

      if(diff > 0)
      {
        if(alpha[j] < 0)
        {
          alpha[j] = 0;
          alpha[i] = diff;
        }
      }
      else
      {
        if(alpha[i] < 0)
        {
          alpha[i] = 0;
          alpha[j] = -diff;
        }
      }
      if(diff > C_i - C_j)
      {
        if(alpha[i] > C_i)
        {
          alpha[i] = C_i;
          alpha[j] = C_i - diff;
        }
      }
      else
      {
        if(alpha[j] > C_j)
        {
          alpha[j] = C_j;
          alpha[i] = C_j + diff;
        }
      }
    }
    else
    {
      double quad_coef = QD[i]+QD[j]-2*Q_i[j];
      if (quad_coef <= 0)
        quad_coef = TAU;
      double delta = (G[i]-G[j])/quad_coef;
      double sum = alpha[i] + alpha[j];
      alpha[i] -= delta;
      alpha[j] += delta;

      if(sum > C_i)
      {
        if(alpha[i] > C_i)
        {
          alpha[i] = C_i;
          alpha[j] = sum - C_i;
        }
      }
      else
      {
        if(alpha[j] < 0)
        {
          alpha[j] = 0;
          alpha[i] = sum;
        }
      }
      if(sum > C_j)
      {
        if(alpha[j] > C_j)
        {
          alpha[j] = C_j;
          alpha[i] = sum - C_j;
        }
      }
      else
      {
        if(alpha[i] < 0)
        {
          alpha[i] = 0;
          alpha[j] = sum;
        }
      }
    }

    // update G

    double delta_alpha_i = alpha[i] - old_alpha_i;
    double delta_alpha_j = alpha[j] - old_alpha_j;

    for(int k=0;k<active_size;k++)
    {
      G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
    }

    // update alpha_status and G_bar

    {
      bool ui = is_upper_bound(i);
      bool uj = is_upper_bound(j);
      update_alpha_status(i);
      update_alpha_status(j);
      int k;
      if(ui != is_upper_bound(i))
      {
        Q_i = Q.get_Q(i,l);
        if(ui)
          for(k=0;k<l;k++)
            G_bar[k] -= C_i * Q_i[k];
        else
          for(k=0;k<l;k++)
            G_bar[k] += C_i * Q_i[k];
      }

      if(uj != is_upper_bound(j))
      {
        Q_j = Q.get_Q(j,l);
        if(uj)
          for(k=0;k<l;k++)
            G_bar[k] -= C_j * Q_j[k];
        else
          for(k=0;k<l;k++)
            G_bar[k] += C_j * Q_j[k];
      }
    }
  }

  if(iter >= max_iter)
  {
    if(active_size < l)
    {
      // reconstruct the whole gradient to calculate objective value
      reconstruct_gradient();
      active_size = l;
      info("*");
    }
    fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
  }

  // calculate rho

  si->rho = calculate_rho();

  // calculate objective value
  {
    double v = 0;
    int i;
    for(i=0;i<l;i++)
      v += alpha[i] * (G[i] + p[i]);

    si->obj = v/2;
  }

  // put back the solution
  {
    for(int i=0;i<l;i++)
      alpha_[active_set[i]] = alpha[i];
  }

  // juggle everything back
  /*{
    for(int i=0;i<l;i++)
      while(active_set[i] != i)
        swap_index(i,active_set[i]);
        // or Q.swap_index(i,active_set[i]);
  }*/

  si->upper_bound_p = Cp;
  si->upper_bound_n = Cn;

  info("\noptimization finished, #iter = %d\n",iter);

  delete[] p;
  delete[] y;
  delete[] alpha;
  delete[] alpha_status;
  delete[] active_set;
  delete[] G;
  delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
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

  for(int t=0;t<active_size;t++)
    if(y[t]==+1)
    {
      if(!is_upper_bound(t))
        if(-G[t] >= Gmax)
        {
          Gmax = -G[t];
          Gmax_idx = t;
        }
    }
    else
    {
      if(!is_lower_bound(t))
        if(G[t] >= Gmax)
        {
          Gmax = G[t];
          Gmax_idx = t;
        }
    }

  int i = Gmax_idx;
  const Qfloat *Q_i = NULL;
  if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
    Q_i = Q->get_Q(i,active_size);

  for(int j=0;j<active_size;j++)
  {
    if(y[j]==+1)
    {
      if (!is_lower_bound(j))
      {
        double grad_diff=Gmax+G[j];
        if (G[j] >= Gmax2)
          Gmax2 = G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          else
            obj_diff = -(grad_diff*grad_diff)/TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
    else
    {
      if (!is_upper_bound(j))
      {
        double grad_diff= Gmax-G[j];
        if (-G[j] >= Gmax2)
          Gmax2 = -G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          else
            obj_diff = -(grad_diff*grad_diff)/TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if(Gmax+Gmax2 < eps)
    return 1;

  out_i = Gmax_idx;
  out_j = Gmin_idx;
  return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2)
{
  if(is_upper_bound(i))
  {
    if(y[i]==+1)
      return(-G[i] > Gmax1);
    else
      return(-G[i] > Gmax2);
  }
  else if(is_lower_bound(i))
  {
    if(y[i]==+1)
      return(G[i] > Gmax2);
    else
      return(G[i] > Gmax1);
  }
  else
    return(false);
}

void Solver::do_shrinking()
{
  int i;
  double Gmax1 = -INF;    // max { -y_i * grad(f)_i | i in I_up(\alpha) }
  double Gmax2 = -INF;    // max { y_i * grad(f)_i | i in I_low(\alpha) }

  // find maximal violating pair first
  for(i=0;i<active_size;i++)
  {
    if(y[i]==+1)
    {
      if(!is_upper_bound(i))
      {
        if(-G[i] >= Gmax1)
          Gmax1 = -G[i];
      }
      if(!is_lower_bound(i))
      {
        if(G[i] >= Gmax2)
          Gmax2 = G[i];
      }
    }
    else
    {
      if(!is_upper_bound(i))
      {
        if(-G[i] >= Gmax2)
          Gmax2 = -G[i];
      }
      if(!is_lower_bound(i))
      {
        if(G[i] >= Gmax1)
          Gmax1 = G[i];
      }
    }
  }

  if(unshrink == false && Gmax1 + Gmax2 <= eps*10)
  {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
    info("*");
  }

  for(i=0;i<active_size;i++)
    if (be_shrunk(i, Gmax1, Gmax2))
    {
      active_size--;
      while (active_size > i)
      {
        if (!be_shrunk(active_size, Gmax1, Gmax2))
        {
          swap_index(i,active_size);
          break;
        }
        active_size--;
      }
    }
}

double Solver::calculate_rho()
{
  double r;
  int nr_free = 0;
  double ub = INF, lb = -INF, sum_free = 0;
  for(int i=0;i<active_size;i++)
  {
    double yG = y[i]*G[i];

    if(is_upper_bound(i))
    {
      if(y[i]==-1)
        ub = min(ub,yG);
      else
        lb = max(lb,yG);
    }
    else if(is_lower_bound(i))
    {
      if(y[i]==+1)
        ub = min(ub,yG);
      else
        lb = max(lb,yG);
    }
    else
    {
      ++nr_free;
      sum_free += yG;
    }
  }

  if(nr_free>0)
    r = sum_free/nr_free;
  else
    r = (ub+lb)/2;

  return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU: public Solver
{
public:
  Solver_NU() {}
  void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
       double *alpha, double Cp, double Cn, double eps,
       SolutionInfo* si, int shrinking)
  {
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
int Solver_NU::select_working_set(int &out_i, int &out_j)
{
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

  for(int t=0;t<active_size;t++)
    if(y[t]==+1)
    {
      if(!is_upper_bound(t))
        if(-G[t] >= Gmaxp)
        {
          Gmaxp = -G[t];
          Gmaxp_idx = t;
        }
    }
    else
    {
      if(!is_lower_bound(t))
        if(G[t] >= Gmaxn)
        {
          Gmaxn = G[t];
          Gmaxn_idx = t;
        }
    }

  int ip = Gmaxp_idx;
  int in = Gmaxn_idx;
  const Qfloat *Q_ip = NULL;
  const Qfloat *Q_in = NULL;
  if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
    Q_ip = Q->get_Q(ip,active_size);
  if(in != -1)
    Q_in = Q->get_Q(in,active_size);

  for(int j=0;j<active_size;j++)
  {
    if(y[j]==+1)
    {
      if (!is_lower_bound(j))
      {
        double grad_diff=Gmaxp+G[j];
        if (G[j] >= Gmaxp2)
          Gmaxp2 = G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          else
            obj_diff = -(grad_diff*grad_diff)/TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
    else
    {
      if (!is_upper_bound(j))
      {
        double grad_diff=Gmaxn-G[j];
        if (-G[j] >= Gmaxn2)
          Gmaxn2 = -G[j];
        if (grad_diff > 0)
        {
          double obj_diff;
          double quad_coef = QD[in]+QD[j]-2*Q_in[j];
          if (quad_coef > 0)
            obj_diff = -(grad_diff*grad_diff)/quad_coef;
          else
            obj_diff = -(grad_diff*grad_diff)/TAU;

          if (obj_diff <= obj_diff_min)
          {
            Gmin_idx=j;
            obj_diff_min = obj_diff;
          }
        }
      }
    }
  }

  if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
    return 1;

  if (y[Gmin_idx] == +1)
    out_i = Gmaxp_idx;
  else
    out_i = Gmaxn_idx;
  out_j = Gmin_idx;

  return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
  if(is_upper_bound(i))
  {
    if(y[i]==+1)
      return(-G[i] > Gmax1);
    else
      return(-G[i] > Gmax4);
  }
  else if(is_lower_bound(i))
  {
    if(y[i]==+1)
      return(G[i] > Gmax2);
    else
      return(G[i] > Gmax3);
  }
  else
    return(false);
}

void Solver_NU::do_shrinking()
{
  double Gmax1 = -INF;  // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
  double Gmax2 = -INF;  // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
  double Gmax3 = -INF;  // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
  double Gmax4 = -INF;  // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

  // find maximal violating pair first
  int i;
  for(i=0;i<active_size;i++)
  {
    if(!is_upper_bound(i))
    {
      if(y[i]==+1)
      {
        if(-G[i] > Gmax1) Gmax1 = -G[i];
      }
      else  if(-G[i] > Gmax4) Gmax4 = -G[i];
    }
    if(!is_lower_bound(i))
    {
      if(y[i]==+1)
      {
        if(G[i] > Gmax2) Gmax2 = G[i];
      }
      else  if(G[i] > Gmax3) Gmax3 = G[i];
    }
  }

  if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10)
  {
    unshrink = true;
    reconstruct_gradient();
    active_size = l;
  }

  for(i=0;i<active_size;i++)
    if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
    {
      active_size--;
      while (active_size > i)
      {
        if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
        {
          swap_index(i,active_size);
          break;
        }
        active_size--;
      }
    }
}

double Solver_NU::calculate_rho()
{
  int nr_free1 = 0,nr_free2 = 0;
  double ub1 = INF, ub2 = INF;
  double lb1 = -INF, lb2 = -INF;
  double sum_free1 = 0, sum_free2 = 0;

  for(int i=0;i<active_size;i++)
  {
    if(y[i]==+1)
    {
      if(is_upper_bound(i))
        lb1 = max(lb1,G[i]);
      else if(is_lower_bound(i))
        ub1 = min(ub1,G[i]);
      else
      {
        ++nr_free1;
        sum_free1 += G[i];
      }
    }
    else
    {
      if(is_upper_bound(i))
        lb2 = max(lb2,G[i]);
      else if(is_lower_bound(i))
        ub2 = min(ub2,G[i]);
      else
      {
        ++nr_free2;
        sum_free2 += G[i];
      }
    }
  }

  double r1,r2;
  if(nr_free1 > 0)
    r1 = sum_free1/nr_free1;
  else
    r1 = (ub1+lb1)/2;

  if(nr_free2 > 0)
    r2 = sum_free2/nr_free2;
  else
    r2 = (ub2+lb2)/2;

  si->r = (r1+r2)/2;
  return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{
public:
  SVC_Q(const Problem& prob, const SVMParameter& param, const schar *y_)
  :Kernel(prob.l, prob.x, param)
  {
    clone(y,y_,prob.l);
    cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
    QD = new double[prob.l];
    for(int i=0;i<prob.l;i++)
      QD[i] = (this->*kernel_function)(i,i);
  }

  Qfloat *get_Q(int i, int len) const
  {
    Qfloat *data;
    int start, j;
    if((start = cache->get_data(i,&data,len)) < len)
    {
      for(j=start;j<len;j++)
        data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
    }
    return data;
  }

  double *get_QD() const
  {
    return QD;
  }

  void swap_index(int i, int j) const
  {
    cache->swap_index(i,j);
    Kernel::swap_index(i,j);
    swap(y[i],y[j]);
    swap(QD[i],QD[j]);
  }

  ~SVC_Q()
  {
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
static void solve_c_svc(
  const Problem *prob, const SVMParameter* param,
  double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
  int l = prob->l;
  double *minus_ones = new double[l];
  schar *y = new schar[l];

  int i;

  for(i=0;i<l;i++)
  {
    alpha[i] = 0;
    minus_ones[i] = -1;
    if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
  }

  Solver s;
  s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
    alpha, Cp, Cn, param->eps, si, param->shrinking);

  double sum_alpha=0;
  for(i=0;i<l;i++)
    sum_alpha += alpha[i];

  if (Cp==Cn)
    info("nu = %f\n", sum_alpha/(Cp*prob->l));

  for(i=0;i<l;i++)
    alpha[i] *= y[i];

  delete[] minus_ones;
  delete[] y;
}

static void solve_nu_svc(
  const Problem *prob, const SVMParameter *param,
  double *alpha, Solver::SolutionInfo* si)
{
  int i;
  int l = prob->l;
  double nu = param->nu;

  schar *y = new schar[l];

  for(i=0;i<l;i++)
    if(prob->y[i]>0)
      y[i] = +1;
    else
      y[i] = -1;

  double sum_pos = nu*l/2;
  double sum_neg = nu*l/2;

  for(i=0;i<l;i++)
    if(y[i] == +1)
    {
      alpha[i] = min(1.0,sum_pos);
      sum_pos -= alpha[i];
    }
    else
    {
      alpha[i] = min(1.0,sum_neg);
      sum_neg -= alpha[i];
    }

  double *zeros = new double[l];

  for(i=0;i<l;i++)
    zeros[i] = 0;

  Solver_NU s;
  s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
    alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
  double r = si->r;

  info("C = %f\n",1/r);

  for(i=0;i<l;i++)
    alpha[i] *= y[i]/r;

  si->rho /= r;
  si->obj /= (r*r);
  si->upper_bound_p = 1/r;
  si->upper_bound_n = 1/r;

  delete[] y;
  delete[] zeros;
}

//
// decision_function
//
struct decision_function
{
  double *alpha;
  double rho;
};

static decision_function svm_train_one(
  const Problem *prob, const SVMParameter *param,
  double Cp, double Cn)
{
  double *alpha = Malloc(double,prob->l);
  Solver::SolutionInfo si;
  switch(param->svm_type)
  {
    case C_SVC:
      solve_c_svc(prob,param,alpha,&si,Cp,Cn);
      break;
    case NU_SVC:
      solve_nu_svc(prob,param,alpha,&si);
      break;
  }

  info("obj = %f, rho = %f\n",si.obj,si.rho);

  // output SVs

  int nSV = 0;
  int nBSV = 0;
  for(int i=0;i<prob->l;i++)
  {
    if(fabs(alpha[i]) > 0)
    {
      ++nSV;
      if(prob->y[i] > 0)
      {
        if(fabs(alpha[i]) >= si.upper_bound_p)
          ++nBSV;
      }
      else
      {
        if(fabs(alpha[i]) >= si.upper_bound_n)
          ++nBSV;
      }
    }
  }

  info("nSV = %d, nBSV = %d\n",nSV,nBSV);

  decision_function f;
  f.alpha = alpha;
  f.rho = si.rho;
  return f;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const Problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
  int l = prob->l;
  int max_nr_class = 16;
  int nr_class = 0;
  int *label = Malloc(int,max_nr_class);
  int *count = Malloc(int,max_nr_class);
  int *data_label = Malloc(int,l);
  int i;

  for(i=0;i<l;i++)
  {
    int this_label = (int)prob->y[i];
    int j;
    for(j=0;j<nr_class;j++)
    {
      if(this_label == label[j])
      {
        ++count[j];
        break;
      }
    }
    data_label[i] = j;
    if(j == nr_class)
    {
      if(nr_class == max_nr_class)
      {
        max_nr_class *= 2;
        label = (int *)realloc(label,(unsigned long)max_nr_class*sizeof(int));
        count = (int *)realloc(count,(unsigned long)max_nr_class*sizeof(int));
      }
      label[nr_class] = this_label;
      count[nr_class] = 1;
      ++nr_class;
    }
  }

  //
  // Labels are ordered by their first occurrence in the training set.
  // However, for two-class sets with -1/+1 labels and -1 appears first,
  // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
  //
  if (nr_class == 2 && label[0] == -1 && label[1] == 1)
  {
    swap(label[0],label[1]);
    swap(count[0],count[1]);
    for(i=0;i<l;i++)
    {
      if(data_label[i] == 0)
        data_label[i] = 1;
      else
        data_label[i] = 0;
    }
  }

  int *start = Malloc(int,nr_class);
  start[0] = 0;
  for(i=1;i<nr_class;i++)
    start[i] = start[i-1]+count[i-1];
  for(i=0;i<l;i++)
  {
    perm[start[data_label[i]]] = i;
    ++start[data_label[i]];
  }
  start[0] = 0;
  for(i=1;i<nr_class;i++)
    start[i] = start[i-1]+count[i-1];

  *nr_class_ret = nr_class;
  *label_ret = label;
  *start_ret = start;
  *count_ret = count;
  free(data_label);
}

//
// Interface functions
//
SVMModel *TrainSVM(const Problem *prob, const SVMParameter *param)
{
  SVMModel *model = Malloc(SVMModel,1);
  model->param = *param;
  model->free_sv = 0;  // XXX

  // classification
  int l = prob->l;
  int nr_class;
  int *label = NULL;
  int *start = NULL;
  int *count = NULL;
  int *perm = Malloc(int,l);

  // group training data of the same class
  svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
  if(nr_class == 1)
    info("WARNING: training data in only one class. See README for details.\n");

  Node **x = Malloc(Node *,l);
  int i;
  for(i=0;i<l;i++)
    x[i] = prob->x[perm[i]];

  // calculate weighted C

  double *weighted_C = Malloc(double, nr_class);
  for(i=0;i<nr_class;i++)
    weighted_C[i] = param->C;
  for(i=0;i<param->nr_weight;i++)
  {
    int j;
    for(j=0;j<nr_class;j++)
      if(param->weight_label[i] == label[j])
        break;
    if(j == nr_class)
      fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
    else
      weighted_C[j] *= param->weight[i];
  }

  // train k*(k-1)/2 models

  bool *nonzero = Malloc(bool,l);
  for(i=0;i<l;i++)
    nonzero[i] = false;
  decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

  int p = 0;
  for(i=0;i<nr_class;i++)
    for(int j=i+1;j<nr_class;j++)
    {
      Problem sub_prob;
      int si = start[i], sj = start[j];
      int ci = count[i], cj = count[j];
      sub_prob.l = ci+cj;
      sub_prob.x = Malloc(Node *,sub_prob.l);
      sub_prob.y = Malloc(double,sub_prob.l);
      int k;
      for(k=0;k<ci;k++)
      {
        sub_prob.x[k] = x[si+k];
        sub_prob.y[k] = +1;
      }
      for(k=0;k<cj;k++)
      {
        sub_prob.x[ci+k] = x[sj+k];
        sub_prob.y[ci+k] = -1;
      }

      f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
      for(k=0;k<ci;k++)
        if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
          nonzero[si+k] = true;
      for(k=0;k<cj;k++)
        if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
          nonzero[sj+k] = true;
      free(sub_prob.x);
      free(sub_prob.y);
      ++p;
    }

  // build output

  model->nr_class = nr_class;

  model->label = Malloc(int,nr_class);
  for(i=0;i<nr_class;i++)
    model->label[i] = label[i];

  model->rho = Malloc(double,nr_class*(nr_class-1)/2);
  for(i=0;i<nr_class*(nr_class-1)/2;i++)
    model->rho[i] = f[i].rho;

  int total_sv = 0;
  int *nz_count = Malloc(int,nr_class);
  model->nSV = Malloc(int,nr_class);
  for(i=0;i<nr_class;i++)
  {
    int nSV = 0;
    for(int j=0;j<count[i];j++)
      if(nonzero[start[i]+j])
      {
        ++nSV;
        ++total_sv;
      }
    model->nSV[i] = nSV;
    nz_count[i] = nSV;
  }

  info("Total nSV = %d\n",total_sv);

  model->l = total_sv;
  model->SV = Malloc(Node *,total_sv);
  model->sv_indices = Malloc(int,total_sv);
  p = 0;
  for(i=0;i<l;i++)
    if(nonzero[i])
    {
      model->SV[p] = x[i];
      model->sv_indices[p++] = perm[i] + 1;
    }

  int *nz_start = Malloc(int,nr_class);
  nz_start[0] = 0;
  for(i=1;i<nr_class;i++)
    nz_start[i] = nz_start[i-1]+nz_count[i-1];

  model->sv_coef = Malloc(double *,nr_class-1);
  for(i=0;i<nr_class-1;i++)
    model->sv_coef[i] = Malloc(double,total_sv);

  p = 0;
  for(i=0;i<nr_class;i++)
    for(int j=i+1;j<nr_class;j++)
    {
      // classifier (i,j): coefficients with
      // i are in sv_coef[j-1][nz_start[i]...],
      // j are in sv_coef[i][nz_start[j]...]

      int si = start[i];
      int sj = start[j];
      int ci = count[i];
      int cj = count[j];

      int q = nz_start[i];
      int k;
      for(k=0;k<ci;k++)
        if(nonzero[si+k])
          model->sv_coef[j-1][q++] = f[p].alpha[k];
      q = nz_start[j];
      for(k=0;k<cj;k++)
        if(nonzero[sj+k])
          model->sv_coef[i][q++] = f[p].alpha[ci+k];
      ++p;
    }

  free(label);
  free(count);
  free(perm);
  free(start);
  free(x);
  free(weighted_C);
  free(nonzero);
  for(i=0;i<nr_class*(nr_class-1)/2;i++)
    free(f[i].alpha);
  free(f);
  free(nz_count);
  free(nz_start);

  return model;
}

int get_svm_type(const SVMModel *model)
{
  return model->param.svm_type;
}

int get_nr_class(const SVMModel *model)
{
  return model->nr_class;
}

void get_labels(const SVMModel *model, int* label)
{
  if (model->label != NULL)
    for(int i=0;i<model->nr_class;i++)
      label[i] = model->label[i];
}

void get_sv_indices(const SVMModel *model, int* indices)
{
  if (model->sv_indices != NULL)
    for(int i=0;i<model->l;i++)
      indices[i] = model->sv_indices[i];
}

int get_nr_sv(const SVMModel *model)
{
  return model->l;
}

double PredictValues(const SVMModel *model, const Node *x, double* dec_values)
{
  int i;
  int nr_class = model->nr_class;
  int l = model->l;

  double *kvalue = Malloc(double,l);
  for(i=0;i<l;i++)
    kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

  int *start = Malloc(int,nr_class);
  start[0] = 0;
  for(i=1;i<nr_class;i++)
    start[i] = start[i-1]+model->nSV[i-1];

  int *vote = Malloc(int,nr_class);
  for(i=0;i<nr_class;i++)
    vote[i] = 0;

  int p=0;
  for(i=0;i<nr_class;i++)
    for(int j=i+1;j<nr_class;j++)
    {
      double sum = 0;
      int si = start[i];
      int sj = start[j];
      int ci = model->nSV[i];
      int cj = model->nSV[j];

      int k;
      double *coef1 = model->sv_coef[j-1];
      double *coef2 = model->sv_coef[i];
      for(k=0;k<ci;k++)
        sum += coef1[si+k] * kvalue[si+k];
      for(k=0;k<cj;k++)
        sum += coef2[sj+k] * kvalue[sj+k];
      sum -= model->rho[p];
      dec_values[p] = sum;

      if(dec_values[p] > 0)
        ++vote[i];
      else
        ++vote[j];
      p++;
    }

  int vote_max_idx = 0;
  for(i=1;i<nr_class;i++)
    if(vote[i] > vote[vote_max_idx])
      vote_max_idx = i;

  free(kvalue);
  free(start);
  free(vote);
  return model->label[vote_max_idx];
}

double PredictSVM(const SVMModel *model, const Node *x)
{
  int nr_class = model->nr_class;
  double *dec_values;
  dec_values = Malloc(double, nr_class*(nr_class-1)/2);
  double pred_result = PredictValues(model, x, dec_values);
  free(dec_values);
  return pred_result;
}

double PredictDecisionValues(const struct SVMModel *model, const struct Node *x, double** dec_values)
{
  int nr_class = model->nr_class;
  *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
  double pred_result = PredictValues(model, x, *dec_values);
  return pred_result;
}

static const char *svm_type_table[] =
{
  "c_svc","nu_svc",NULL
};

static const char *kernel_type_table[]=
{
  "linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int SaveSVMModel(const char *model_file_name, const SVMModel *model)
{
  FILE *fp = fopen(model_file_name,"w");
  if(fp==NULL) return -1;

  char *old_locale = strdup(setlocale(LC_ALL, NULL));
  setlocale(LC_ALL, "C");

  const SVMParameter& param = model->param;

  fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
  fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

  if(param.kernel_type == POLY)
    fprintf(fp,"degree %d\n", param.degree);

  if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
    fprintf(fp,"gamma %g\n", param.gamma);

  if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
    fprintf(fp,"coef0 %g\n", param.coef0);

  int nr_class = model->nr_class;
  int l = model->l;
  fprintf(fp, "nr_class %d\n", nr_class);
  fprintf(fp, "total_sv %d\n",l);

  {
    fprintf(fp, "rho");
    for(int i=0;i<nr_class*(nr_class-1)/2;i++)
      fprintf(fp," %g",model->rho[i]);
    fprintf(fp, "\n");
  }

  if(model->label)
  {
    fprintf(fp, "label");
    for(int i=0;i<nr_class;i++)
      fprintf(fp," %d",model->label[i]);
    fprintf(fp, "\n");
  }

  if(model->nSV)
  {
    fprintf(fp, "nr_sv");
    for(int i=0;i<nr_class;i++)
      fprintf(fp," %d",model->nSV[i]);
    fprintf(fp, "\n");
  }

  fprintf(fp, "SV\n");
  const double * const *sv_coef = model->sv_coef;
  const Node * const *SV = model->SV;

  for(int i=0;i<l;i++)
  {
    for(int j=0;j<nr_class-1;j++)
      fprintf(fp, "%.16g ",sv_coef[j][i]);

    const Node *p = SV[i];

    if(param.kernel_type == PRECOMPUTED)
      fprintf(fp,"0:%d ",(int)(p->value));
    else
      while(p->index != -1)
      {
        fprintf(fp,"%d:%.8g ",p->index,p->value);
        p++;
      }
    fprintf(fp, "\n");
  }

  setlocale(LC_ALL, old_locale);
  free(old_locale);

  if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
  else return 0;
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
  int len;

  if(fgets(line,max_line_len,input) == NULL)
    return NULL;

  while(strrchr(line,'\n') == NULL)
  {
    max_line_len *= 2;
    line = (char *) realloc(line,(size_t)max_line_len);
    len = (int) strlen(line);
    if(fgets(line+len,max_line_len-len,input) == NULL)
      break;
  }
  return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return false; }while(0)
bool read_model_header(FILE *fp, SVMModel* model)
{
  SVMParameter& param = model->param;
  char cmd[81];
  while(1)
  {
    FSCANF(fp,"%80s",cmd);

    if(strcmp(cmd,"svm_type")==0)
    {
      FSCANF(fp,"%80s",cmd);
      int i;
      for(i=0;svm_type_table[i];i++)
      {
        if(strcmp(svm_type_table[i],cmd)==0)
        {
          param.svm_type=i;
          break;
        }
      }
      if(svm_type_table[i] == NULL)
      {
        fprintf(stderr,"unknown svm type.\n");
        return false;
      }
    }
    else if(strcmp(cmd,"kernel_type")==0)
    {
      FSCANF(fp,"%80s",cmd);
      int i;
      for(i=0;kernel_type_table[i];i++)
      {
        if(strcmp(kernel_type_table[i],cmd)==0)
        {
          param.kernel_type=i;
          break;
        }
      }
      if(kernel_type_table[i] == NULL)
      {
        fprintf(stderr,"unknown kernel function.\n");
        return false;
      }
    }
    else if(strcmp(cmd,"degree")==0)
      FSCANF(fp,"%d",&param.degree);
    else if(strcmp(cmd,"gamma")==0)
      FSCANF(fp,"%lf",&param.gamma);
    else if(strcmp(cmd,"coef0")==0)
      FSCANF(fp,"%lf",&param.coef0);
    else if(strcmp(cmd,"nr_class")==0)
      FSCANF(fp,"%d",&model->nr_class);
    else if(strcmp(cmd,"total_sv")==0)
      FSCANF(fp,"%d",&model->l);
    else if(strcmp(cmd,"rho")==0)
    {
      int n = model->nr_class * (model->nr_class-1)/2;
      model->rho = Malloc(double,n);
      for(int i=0;i<n;i++)
        FSCANF(fp,"%lf",&model->rho[i]);
    }
    else if(strcmp(cmd,"label")==0)
    {
      int n = model->nr_class;
      model->label = Malloc(int,n);
      for(int i=0;i<n;i++)
        FSCANF(fp,"%d",&model->label[i]);
    }
    else if(strcmp(cmd,"nr_sv")==0)
    {
      int n = model->nr_class;
      model->nSV = Malloc(int,n);
      for(int i=0;i<n;i++)
        FSCANF(fp,"%d",&model->nSV[i]);
    }
    else if(strcmp(cmd,"SV")==0)
    {
      while(1)
      {
        int c = getc(fp);
        if(c==EOF || c=='\n') break;
      }
      break;
    }
    else
    {
      fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
      return false;
    }
  }

  return true;

}

SVMModel *LoadSVMModel(const char *model_file_name)
{
  FILE *fp = fopen(model_file_name,"rb");
  if(fp==NULL) return NULL;

  char *old_locale = strdup(setlocale(LC_ALL, NULL));
  setlocale(LC_ALL, "C");

  // read parameters

  SVMModel *model = Malloc(SVMModel,1);
  model->rho = NULL;
  model->sv_indices = NULL;
  model->label = NULL;
  model->nSV = NULL;

  // read header
  if (!read_model_header(fp, model))
  {
    fprintf(stderr, "ERROR: fscanf failed to read model\n");
    setlocale(LC_ALL, old_locale);
    free(old_locale);
    free(model->rho);
    free(model->label);
    free(model->nSV);
    free(model);
    return NULL;
  }

  // read sv_coef and SV

  int elements = 0;
  long pos = ftell(fp);

  max_line_len = 1024;
  line = Malloc(char,max_line_len);
  char *p,*endptr,*idx,*val;

  while(readline(fp)!=NULL)
  {
    p = strtok(line,":");
    while(1)
    {
      p = strtok(NULL,":");
      if(p == NULL)
        break;
      ++elements;
    }
  }
  elements += model->l;

  fseek(fp,pos,SEEK_SET);

  int m = model->nr_class - 1;
  int l = model->l;
  model->sv_coef = Malloc(double *,m);
  int i;
  for(i=0;i<m;i++)
    model->sv_coef[i] = Malloc(double,l);
  model->SV = Malloc(Node*,l);
  Node *x_space = NULL;
  if(l>0) x_space = Malloc(Node,elements);

  int j=0;
  for(i=0;i<l;i++)
  {
    readline(fp);
    model->SV[i] = &x_space[j];

    p = strtok(line, " \t");
    model->sv_coef[0][i] = strtod(p,&endptr);
    for(int k=1;k<m;k++)
    {
      p = strtok(NULL, " \t");
      model->sv_coef[k][i] = strtod(p,&endptr);
    }

    while(1)
    {
      idx = strtok(NULL, ":");
      val = strtok(NULL, " \t");

      if(val == NULL)
        break;
      x_space[j].index = (int) strtol(idx,&endptr,10);
      x_space[j].value = strtod(val,&endptr);

      ++j;
    }
    x_space[j++].index = -1;
  }
  free(line);

  setlocale(LC_ALL, old_locale);
  free(old_locale);

  if (ferror(fp) != 0 || fclose(fp) != 0)
    return NULL;

  model->free_sv = 1;  // XXX
  return model;
}

void svm_free_model_content(SVMModel* model_ptr)
{
  if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
    free((void *)(model_ptr->SV[0]));
  if(model_ptr->sv_coef)
  {
    for(int i=0;i<model_ptr->nr_class-1;i++)
      free(model_ptr->sv_coef[i]);
  }

  if (model_ptr->SV)
  {
    free(model_ptr->SV);
    model_ptr->SV = NULL;
  }

  if (model_ptr->sv_coef)
  {
    free(model_ptr->sv_coef);
    model_ptr->sv_coef = NULL;
  }

  if (model_ptr->rho)
  {
    free(model_ptr->rho);
    model_ptr->rho = NULL;
  }

  if (model_ptr->label)
  {
    free(model_ptr->label);
    model_ptr->label= NULL;
  }

  if (model_ptr->sv_indices)
  {
    free(model_ptr->sv_indices);
    model_ptr->sv_indices = NULL;
  }

  if (model_ptr->nSV)
  {
    free(model_ptr->nSV);
    model_ptr->nSV = NULL;
  }
}

void FreeSVMModel(SVMModel** model_ptr_ptr)
{
  if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
  {
    svm_free_model_content(*model_ptr_ptr);
    free(*model_ptr_ptr);
    *model_ptr_ptr = NULL;
  }
}

void FreeSVMParam(SVMParameter* param)
{
  free(param->weight_label);
  free(param->weight);
}

const char *CheckSVMParameter(const Problem *prob, const SVMParameter *param)
{
  // svm_type

  int svm_type = param->svm_type;
  if(svm_type != C_SVC &&
     svm_type != NU_SVC)
    return "unknown svm type";

  // kernel_type, degree

  int kernel_type = param->kernel_type;
  if(kernel_type != LINEAR &&
     kernel_type != POLY &&
     kernel_type != RBF &&
     kernel_type != SIGMOID &&
     kernel_type != PRECOMPUTED)
    return "unknown kernel type";

  if(param->gamma < 0)
    return "gamma < 0";

  if(param->degree < 0)
    return "degree of polynomial kernel < 0";

  // cache_size,eps,C,nu,p,shrinking

  if(param->cache_size <= 0)
    return "cache_size <= 0";

  if(param->eps <= 0)
    return "eps <= 0";

  if(svm_type == C_SVC)
    if(param->C <= 0)
      return "C <= 0";

  if(svm_type == NU_SVC)
    if(param->nu <= 0 || param->nu > 1)
      return "nu <= 0 or nu > 1";

  if(param->shrinking != 0 &&
     param->shrinking != 1)
    return "shrinking != 0 and shrinking != 1";

  // check whether nu-svc is feasible

  if(svm_type == NU_SVC)
  {
    int l = prob->l;
    int max_nr_class = 16;
    int nr_class = 0;
    int *label = Malloc(int,max_nr_class);
    int *count = Malloc(int,max_nr_class);

    int i;
    for(i=0;i<l;i++)
    {
      int this_label = (int)prob->y[i];
      int j;
      for(j=0;j<nr_class;j++)
        if(this_label == label[j])
        {
          ++count[j];
          break;
        }
      if(j == nr_class)
      {
        if(nr_class == max_nr_class)
        {
          max_nr_class *= 2;
          label = (int *)realloc(label,(unsigned long)max_nr_class*sizeof(int));
          count = (int *)realloc(count,(unsigned long)max_nr_class*sizeof(int));
        }
        label[nr_class] = this_label;
        count[nr_class] = 1;
        ++nr_class;
      }
    }

    for(i=0;i<nr_class;i++)
    {
      int n1 = count[i];
      for(int j=i+1;j<nr_class;j++)
      {
        int n2 = count[j];
        if(param->nu*(n1+n2)/2 > min(n1,n2))
        {
          free(label);
          free(count);
          return "specified nu is infeasible";
        }
      }
    }
    free(label);
    free(count);
  }

  return NULL;
}

void svm_set_print_string_function(void (*print_func)(const char *))
{
  if(print_func == NULL)
    svm_print_string = &print_string_stdout;
  else
    svm_print_string = print_func;
}
