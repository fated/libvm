#include "utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char const *argv[])
{
  read_problem("iris_3");
  return 0;
}

void read_problem(const char *filename)
{
  std::string line;
  std::ifstream file(filename);
  std::vector<std::string> tokens;
  int elements, max_index;
  problem prob;

  if (!file.is_open()) {
    std::cerr << "Unable to open input file " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  prob.l = 0;
  elements = 0;

  while (std::getline(file, line)) {
    std::size_t prev = 0, pos;
    pos = line.find_first_of(" \t", prev);
    prev = pos + 1;
    while ((pos = line.find_first_of(" \t", prev)) != std::string::npos) {
      std::cout << pos << ' ';
      if (pos > prev)
        ++elements;
      prev = pos + 1;
    }
    ++elements;
    ++prob.l;
  }
  file.seekg(0);

  prob.y = new double[prob.l];
  prob.x = new node*[prob.l];
  // while (std::getline(file, line)) {
  //   std::size_t prev = 0, pos;
  //   while ((pos = line.find_first_of(" \t", prev)) != std::string::npos) {
  //     if (pos > prev)
  //       tokens.push_back(line.substr(prev, pos-prev));
  //     prev = pos + 1;
  //   }
  //   if (prev < line.length())
  //     tokens.push_back(line.substr(prev, std::string::npos));
  // }

  file.close();
}