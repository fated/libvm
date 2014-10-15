#include "mcsvm.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

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