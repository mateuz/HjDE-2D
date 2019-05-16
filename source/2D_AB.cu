#include "2D_AB.cuh"
#include "IO.h"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

F2DAB::F2DAB( uint _dim, uint _ps ):Benchmarks()
{
  //protein size
  n_dim = _dim;

  //number of individuals
  ps = _ps;

  min = -3.1415926535897932384626433832795029;
  max = +3.1415926535897932384626433832795029;

  ID = 1001;

  // get the next multiple of 32;
  // n_threads = 32 * ceil((double) n_dim / 32.0);

  //one block per population member
  // n_blocks = ps;

  n_threads = 128;
  n_blocks = ps;

  printf("nb: %d e nt: %d\n", n_blocks, n_threads);

  char s_2dab[150];
  memset(s_2dab, 0, sizeof(char) * 150);

  if( n_dim == 13 ){
    strcpy(s_2dab, "ABBABBABABBAB");
  } else if( n_dim == 21 ){
    strcpy(s_2dab, "BABABBABABBABBABABBAB");
  } else if( n_dim == 34 ){
    strcpy(s_2dab, "ABBABBABABBABBABABBABABBABBABABBAB");
  } else if( n_dim == 38 ){
    strcpy(s_2dab, "AAAABABABABABAABAABBAAABBABAABBBABABAB");
  } else if( n_dim == 55 ){
    strcpy(s_2dab, "BABABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBAB");
  } else if( n_dim == 64 ){
    strcpy(s_2dab, "ABBABAABBABABBBAABBABABBBABBABABBABABBABABABAABABBAABBABBBAAABAB");
  } else if( n_dim == 98 ){
    strcpy(s_2dab, "AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA");
  } else if( n_dim == 120 ){
    strcpy(s_2dab, "ABBABBAABABABAABBAAAABAABABBABABBAAABBBAABBBABAAABABBABBABBBBABBBBAABBBBBBBABABBAAAABBBBBBABBBBAAAABBBABABBBBAAAABBABABB");
  } else {
    std::cout << "error string size must be 13, 21, 34, 38, 55, 64, 98, and 120.\n";
    exit(-1);
  }

  checkCudaErrors(cudaMemcpyToSymbol(S_2DAB, (void *) s_2dab, 150 * sizeof(char)));
}

F2DAB::~F2DAB()
{
  /* empty */
}

__device__ float _C( uint i, uint j ){
  float c;

  if( S_2DAB[i] == 'A' && S_2DAB[j] == 'A' )
    c = 1.0;
  else if( S_2DAB[i] == 'B' && S_2DAB[j] == 'B' )
    c = 0.5;
  else
    c = -0.5;

  return c;
}

__global__ void computeK_2DAB(float * x, float * f){
  uint id_p = blockIdx.x;
  uint id_d = threadIdx.x;
  uint ndim = params.n_dim;

  uint stride = id_p * ndim;

  // if( id_p == 0 && id_d == 0 ){
  //   printf("Otimizando a string: %s\n", S_2DAB);
  //   printf("Nº de dimensões: %d\n", params.n_dim);
  //   printf("Nº de Indivíduos: %d\n", params.ps);
  //   printf("x in [%.3f, %.3f]\n", params.x_min, params.x_max);
  // }

  __shared__ AB_2D amino_acid[150];

  double d_x, d_y;

  if( id_d == 0 ){
    // printf("STRIDE for block %d is %d\n", id_p, stride);
    amino_acid[0].x = 0.0;
    amino_acid[0].y = 0.0;

    amino_acid[1].x = 1.0;
    amino_acid[1].y = 0.0;

    for( int i = 1; i < (ndim - 1); i++ ){
      d_x = amino_acid[i].x - amino_acid[i-1].x;
      d_y = amino_acid[i].y - amino_acid[i-1].y;

      amino_acid[i+1].x = amino_acid[i].x + d_x * cosf( x[stride + i - 1]) - d_y * sinf( x[stride + i - 1] );
      amino_acid[i+1].y = amino_acid[i].y + d_y * cosf( x[stride + i - 1]) + d_x * sinf( x[stride + i - 1] );
    }
  }

  __shared__ float v1[128], v2[128];

  v1[id_d] = 0.0;
  v2[id_d] = 0.0;

  __syncthreads();

  float C, D;
  if( id_d < (ndim - 2) ){
    v1[id_d] = (1.0 - cosf(x[stride + id_d])) / 4.0f;

    float _v2 = 0.0;
    for( uint j = (id_d+2); j < ndim; j++ ){
      C = _C(id_d, j);

      d_x = amino_acid[id_d].x - amino_acid[j].x;
      d_y = amino_acid[id_d].y - amino_acid[j].y;

      D = sqrtf( (d_x * d_x) + (d_y * d_y) );
      _v2 += 4.0 * ( 1/powf(D, 12) - C/powf(D, 6) );
    }
    v2[id_d] += _v2;
  }

  // Just apply a simple reduce sum to get the v1 and v2 sum
  // if( id_d < 128 && ndim > 128 ){
  //   v1[id_d] += v1[id_d + 128];
  //   v2[id_d] += v2[id_d + 128];
  // }

  __syncthreads();

  if( id_d < 64 && ndim > 64 ){
    v1[id_d] += v1[id_d + 64];
    v2[id_d] += v2[id_d + 64];
  }

  __syncthreads();

  if( id_d < 32 && ndim > 32 ){
    v1[id_d] += v1[id_d + 32];
    v2[id_d] += v2[id_d + 32];
  }

  __syncthreads();

  if( id_d < 16 && ndim > 16 ){
    v1[id_d] += v1[id_d + 16];
    v2[id_d] += v2[id_d + 16];
  }

  __syncthreads();

  if( id_d < 8 ){
    v1[id_d] += v1[id_d + 8];
    v2[id_d] += v2[id_d + 8];
  }

  __syncthreads();

  if( id_d < 4 ){
    v1[id_d] += v1[id_d + 4];
    v2[id_d] += v2[id_d + 4];
  }

  __syncthreads();

  if( id_d < 2 ){
    v1[id_d] += v1[id_d + 2];
    v2[id_d] += v2[id_d + 2];
  }

  __syncthreads();

  if( id_d == 0 ){
    v1[id_d] += v1[id_d + 1];
    v2[id_d] += v2[id_d + 1];

    f[id_p] = v1[0] + v2[0];
    // printf("[%d] %.3lf from %.3lf and %.3lf\n", id_p, f[id_p], v1[0], v2[0]);
  }
}

__global__ void computeS(float *x, float *f){
   uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
   uint ps = params.ps;

   if( id_p < ps ){
     uint ndim = params.n_dim;
     uint id_d = id_p * ndim;

     AB_2D amino_acid[150];

     unsigned int i = 0, j = 0;
     float a_ab,b_ab,c_ab,d_ab,v1,v2;

   	 amino_acid[0].x = 0;
     amino_acid[0].y = 0;
   	 amino_acid[1].x = 1;
   	 amino_acid[1].y = 0;

   	for (i = 1; i < (ndim - 1); i++){
   		a_ab = amino_acid[i].x-amino_acid[i-1].x;
   		b_ab = amino_acid[i].y-amino_acid[i-1].y;
   		amino_acid[i+1].x = amino_acid[i].x+a_ab*cosf(x[id_d + i - 1])-b_ab*sinf(x[id_d + i - 1]);
   		amino_acid[i+1].y = amino_acid[i].y+b_ab*cosf(x[id_d + i - 1])+a_ab*sinf(x[id_d + i - 1]);
   	}
    v1 = 0.0;
    v2 = 0.0;
    for (i = 0; (i < (ndim-2)); i++) {
      v1 += (1.0 - cosf(x[id_d + i]) ) / 4.0;
      for (j = (i+2); (j < ndim); j++) {
        if (S_2DAB[i] == 'A' && S_2DAB[j] == 'A') //AA bond
          c_ab = 1;
        else if (S_2DAB[i] == 'B' && S_2DAB[j] == 'B') //BB bond
          c_ab = 0.5;
        else
          c_ab = -0.5; //AB or BA bond

        d_ab = sqrtf(((amino_acid[i].x-amino_acid[j].x)*(amino_acid[i].x-amino_acid[j].x))+((amino_acid[i].y-amino_acid[j].y)*(amino_acid[i].y-amino_acid[j].y))); //Distance for Lennard-Jones
        v2 += 4.0 * ( 1 / powf(d_ab, 12) - c_ab / powf(d_ab, 6) );
      }
    }
    f[id_p] = v1 + v2;
   }
}

void F2DAB::compute(float * x, float * f){
  // computeK_2DAB<<< n_blocks, n_threads >>>(x, f);
  computeK_2DAB<<< 200, 128 >>>(x, f);
  // computeS<<<20, 10>>>(x,f);
  // cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
