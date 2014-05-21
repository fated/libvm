#ifndef LIBVM_UTILITIES_H_
#define LIBVM_UTILITIES_H_

#include <vector>
#include <map>

#define INF HUGE_VAL
#define TAU 1e-12

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

template<typename T>
T FindMostFrequent(T *array, int size)
{
  std::vector<T> v(array, array+size);
  std::map<T, int> frequency_map;
  int max_frequency = 0;
  T most_frequent_element;

  for (std::size_t i = 0; i != v.size(); ++i) {
    int cur_frequency = ++frequency_map[v[i]];
    if (cur_frequency > max_frequency) {
        max_frequency = cur_frequency;
        most_frequent_element = v[i];
    }
  }

  return most_frequent_element;
}

struct Problem *ReadProblem(const char *file_name);
void FreeProblem(struct Problem *problem);

#endif  // LIBVM_UTILITIES_H_
