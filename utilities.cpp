#include "utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <exception>

struct Problem *ReadProblem(const char *file_name)
{
  std::string line;
  std::ifstream input_file(file_name);
  int max_index, current_max_index;
  std::size_t elements;
  Problem *problem = new Problem;

  if (!input_file.is_open()) {
    std::cerr << "Unable to open input file: " << file_name << std::endl;
    exit(EXIT_FAILURE);
  }

  problem->l = 0;
  elements = 0;

  while (std::getline(input_file, line)) {
    ++problem->l;
  }
  input_file.clear();
  input_file.seekg(0);

  problem->y = new double[problem->l];
  problem->x = new Node*[problem->l];

  max_index = 0;
  for (int i = 0; i < problem->l; ++i) {
    std::vector<std::string> tokens;
    std::size_t prev = 0, pos;

    current_max_index = -1;
    std::getline(input_file, line);
    while ((pos = line.find_first_of(" \t\n", prev)) != std::string::npos) {
      if (pos > prev)
        tokens.push_back(line.substr(prev, pos-prev));
      prev = pos + 1;
    }
    if (prev < line.length())
      tokens.push_back(line.substr(prev, std::string::npos));

    try
    {
      std::size_t end;

      problem->y[i] = std::stod(tokens[0], &end);
      if (end != tokens[0].length()) {
        throw std::invalid_argument("incomplete convention");
      }
    }
    catch(std::exception& e)
    {
      std::cerr << "Error: " << e.what() << " in line " << (i+1) << std::endl;
      delete[] problem->y;
      for (int j = 0; j < i; ++j) {
        delete[] problem->x[j];
      }
      delete[] problem->x;
      std::vector<std::string>(tokens).swap(tokens);
      exit(EXIT_FAILURE);
    }  // TODO try not to use exception

    elements = tokens.size();
    problem->x[i] = new Node[elements];
    prev = 0;
    for (std::size_t j = 0; j < elements-1; ++j) {
      pos = tokens[j+1].find_first_of(':');
      try
      {
        std::size_t end;

        problem->x[i][j].index = std::stoi(tokens[j+1].substr(prev, pos-prev), &end);
        if (end != (tokens[j+1].substr(prev, pos-prev)).length()) {
          throw std::invalid_argument("incomplete convention");
        }
        problem->x[i][j].value = std::stod(tokens[j+1].substr(pos+1), &end);
        if (end != (tokens[j+1].substr(pos+1)).length()) {
          throw std::invalid_argument("incomplete convention");
        }
      }
      catch(std::exception& e)
      {
        std::cerr << "Error: " << e.what() << " in line " << (i+1) << std::endl;
        delete[] problem->y;
        for (int j = 0; j < i+1; ++j) {
          delete[] problem->x[j];
        }
        delete[] problem->x;
        std::vector<std::string>(tokens).swap(tokens);
        exit(EXIT_FAILURE);
      }
      current_max_index = problem->x[i][j].index;
    }

    if (current_max_index > max_index) {
      max_index = current_max_index;
    }
    problem->x[i][elements-1].index = -1;
    problem->x[i][elements-1].value = 0;
  }

  problem->max_index = max_index;

  // TODO add precomputed kernel check

  input_file.close();
  return problem;
}

void FreeProblem(struct Problem *problem)
{
  delete[] problem->y;
  for (int i = 0; i < problem->l; ++i) {
    delete[] problem->x[i];
  }
  delete[] problem->x;
  delete problem;

  return;
}