#ifndef LIBVM_KNN_H_
#define LIBVM_KNN_H_

struct KNNParameter
{
  int num_neighbors;
};

double KNN(struct Problem *train, struct Node *x, const int num_neighbors);
double CalcDist(struct Node *x1, struct Node *x2);
int CompareDist(double *neighbors, double dist, int num_neighbors);
void InsertLabel(double *labels, double label, int num_neighbors, int index);

#endif  // LIBVM_KNN_H_
