#ifndef __jDE__
#define __jDE__

#include "helper.cuh"
#include "constants.cuh"

/* C++ includes */
#include <tuple>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>
#include <random>

class jDE {
protected:
  uint NP;
  uint n_dim;

  dim3 NT_A;
  dim3 NB_A;

  dim3 NT_B;
  dim3 NB_B;

  float x_min;
  float x_max;

  /* device data */
  curandState * d_states;
  curandState * d_states2;

  uint * rseq;
  uint * fseq;
  float * F;
  float * CR;

  float * T_F;
  float * T_CR;

public:
  jDE( uint, uint, float, float );
  ~jDE();

  /* jDE functions */
  void run(float *, float *);
  void run_b(float *, float *, float *, float *, float *, uint);
  void update();
  void selection(float *, float *, float *, float *);
  void crowding_selection(float *, float *, float *, float *, float *);
  void index_gen();
  void reset();
};

/* CUDA Kernels */
__global__ void updateK(curandState *, float *, float *, float *, float *);

__global__ void selectionK(float *, float *, float *, float *);

__global__ void DE(curandState *, float *, float *, float *, float *, uint *);

__global__ void rand_DE(curandState *, float *, float *, float *, float *, uint *);

__global__ void best_DE(float *, float *, float *, float *, float *, uint);

__global__ void crowding(float *, float *, uint, float *);

__global__ void iGen(curandState *, uint *, uint *);

__global__ void setup_kernel(curandState *, uint);

__global__ void sk2(curandState *, uint);

#endif
