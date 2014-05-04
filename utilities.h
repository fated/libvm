#ifndef LIBVM_UTILITIES_H_
#define LIBVM_UTILITIES_H_

struct node
{
  int index;
  double value;
};

struct problem
{
  int l;
  int max_index;
  double *y;
  struct node **x;
};

void read_problem(const char *filename);

#endif  // LIBVM_UTILITIES_H_
