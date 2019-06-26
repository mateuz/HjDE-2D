#ifndef _3DAB_H
#define _3DAB_H

#include "Benchmarks.cuh"
#include "helper.cuh"

class F3DAB : public Benchmarks
{
private:
  /* empty */

public:
  F3DAB( uint, uint, uint );
  ~F3DAB();

  void compute(float * x, float * fitness);

};

__global__ void computeK_3DAB_P(float * x, float * f);
__global__ void computeK_3DAB_S(float * x, float * f);

#endif
