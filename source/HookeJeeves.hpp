#ifndef _HOOKEJEEVES_H
#define _HOOKEJEEVES_H

#include <tuple>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>

typedef struct {
	float x;
	float y;
} amino;

class HookeJeeves
{
public:

  // this is the number of dimensions
  uint nvars;

  // this is the user-supplied guess at the minimum
  float * startpt;

  // this is the localtion of the local minimum, calculated by the program
  float * endpt;

  // this is to control the perturbation in each dimension
  float * delta;

  // aux vectors
  float * newx;
  float * xbef;
  float * z;

  // this is a user-supplied convergence parameter,
  // which should be set to a value between 0.0 and 1.0.
  // Larger	values of rho give greater probability of
  // convergence on highly nonlinear functions, at a
  // cost of more function evaluations.  Smaller values
  // of rho reduces the number of evaluations (and the
  // program running time), but increases the risk of
  // nonconvergence.
  float rho;

  // this is the criterion for halting the search for a minimum.
  float epsilon;

  // A second, rarely used, halting criterion. If the algorithm
  // uses >= itermax iterations, halt.
  uint itermax;

  amino * amino_pos;

  std::string AB_SQ;

  // Parameters received:
  //   - uint: number of Dimensions
  //   - float: rho
  //   - float: epsilon
	HookeJeeves(uint, float, float);
	~HookeJeeves();

  float best_nearby(float *, float , uint * );
  float optimize(const uint, float *);
  float evaluate(float *);
};

#endif
