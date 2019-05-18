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
	double x;
	double y;
} amino;

#define PI 3.1415926535897932384626433832795029

class HookeJeeves
{
public:

  // this is the number of dimensions
  uint nvars;

  // this is the user-supplied guess at the minimum
  double * startpt;

  // this is the localtion of the local minimum, calculated by the program
  double * endpt;

  // this is to control the perturbation in each dimension
  double * delta;

  // aux vectors
  double * newx;
  double * xbef;
  double * z;

  // this is a user-supplied convergence parameter,
  // which should be set to a value between 0.0 and 1.0.
  // Larger	values of rho give greater probability of
  // convergence on highly nonlinear functions, at a
  // cost of more function evaluations.  Smaller values
  // of rho reduces the number of evaluations (and the
  // program running time), but increases the risk of
  // nonconvergence.
  double rho;

  // this is the criterion for halting the search for a minimum.
  double epsilon;

  // A second, rarely used, halting criterion. If the algorithm
  // uses >= itermax iterations, halt.
  uint itermax;

  amino * amino_pos;

  std::string AB_SQ;

  // Parameters received:
  //   - uint: number of Dimensions
  //   - double: rho
  //   - double: epsilon
	HookeJeeves(uint, double, double);
	~HookeJeeves();

  double best_nearby(double *, double , uint * );
  double optimize(const uint, double *);
  double evaluate(const double *);
};

#endif
