#include "utilities.h"
#include <string>

int main(int argc, char *argv[])
{
  char *input_file_name = argv[1];
  struct Problem *prob;

  prob = ReadProblem(input_file_name);
  return 0;
}