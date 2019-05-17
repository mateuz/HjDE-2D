#ifndef _2DAB_H
#define _2DAB_H

#include "Benchmarks.cuh"
#include "helper.cuh"

typedef struct {
  float x, y;
} AB_2D;

class F2DAB : public Benchmarks
{
private:
  /* empty */

public:
  F2DAB( uint, uint );
  ~F2DAB();

  void compute(float * x, float * fitness);

};

__device__ float _C( uint i, uint j );

__global__ void computeK_2DAB_P(float * x, float * f);
__global__ void computeK_2DAB_S(float * x, float * f);

#endif
