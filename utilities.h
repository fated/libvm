#ifndef LIBVM_UTILITIES_H_
#define LIBVM_UTILITIES_H_

struct Node
{
  int index;
  double value;
};

struct Problem
{
  int l;  // number of examples
  int max_index;
  double *y;
  struct Node **x;
};

struct Problem *ReadProblem(const char *file_name);
void test();

#endif  // LIBVM_UTILITIES_H_
