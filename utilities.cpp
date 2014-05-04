#include "utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <exception>

int main(int argc, char const *argv[])
{
  read_problem("iris_3");
  return 0;
}

void read_problem(const char *filename)
{
  std::string line;
  std::ifstream file(filename);
  int elements, max_index, inst_max_index;
  problem prob;

  if (!file.is_open()) {
    std::cerr << "Unable to open input file " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  prob.l = 0;
  elements = 0;

  while (std::getline(file, line)) {
    ++prob.l;
  }
  file.clear();
  file.seekg(0);

  prob.y = new double[prob.l];
  prob.x = new node*[prob.l];

  max_index = 0;
  for (int i = 0; i < prob.l; ++i) {
    std::vector<std::string> tokens;

    inst_max_index = -1;
    std::getline(file, line);
    std::size_t prev = 0, pos;
    while ((pos = line.find_first_of(" \t\n", prev)) != std::string::npos) {
      if (pos > prev)
        tokens.push_back(line.substr(prev, pos-prev));
      prev = pos + 1;
    }
    if (prev < line.length())
      tokens.push_back(line.substr(prev, std::string::npos));

    try
    {
      prob.y[i] = stod(tokens[0]);
    }
    catch(std::exception& e)
    {
      std::cerr << "Error: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    elements = tokens.size();
    prob.x[i] = new node[elements];
    prev = 0;
    for (int j = 0; j < elements-1; ++j) {
      pos = tokens[j+1].find_first_of(':');
      prob.x[i][j].index = stoi(tokens[j+1].substr(prev, pos-prev));
      prob.x[i][j].value = stod(tokens[j+1].substr(pos+1));
      std::cout << prob.x[i][j].index << ' ' << prob.x[i][j].value << std::endl;
    }
  }

  file.close();
}