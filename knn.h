#ifndef LIBVM_KNN_H_
#define LIBVM_KNN_H_

struct KNNParameter
{
  int num_neighbors;
};

double KNN(struct Problem *train, struct Node *x, const int num_neighbors);

#endif  // LIBVM_KNN_H_
