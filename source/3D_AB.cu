#include "3D_AB.cuh"
#include "constants.cuh"

#include <iostream>
#include <vector>
#include <iterator>
#include <fstream>

F3DAB::F3DAB( uint _dim, uint _ps, uint _pl ):Benchmarks()
{
  //dimensions to opt
  n_dim = _dim;

  //protein Length
  protein_length = _pl;

  //number of individuals
  ps = _ps;

  min = -3.1415926535897932384626433832795029;
  max = +3.1415926535897932384626433832795029;

  ID = 1002;

  // get the next multiple of 32;
  NT.x = 32 * ceil((double) protein_length / 32.0);

  //one block per population member
  NB.x = ps;

  // printf("nb: %d e nt: %d\n", n_blocks, n_threads);

  char s_2dab[150];
  memset(s_2dab, 0, sizeof(char) * 150);

  if( protein_length == 13 ){
    strcpy(s_2dab, "ABBABBABABBAB");
  } else if( protein_length == 21 ){
    strcpy(s_2dab, "BABABBABABBABBABABBAB");
  } else if( protein_length == 34 ){
    strcpy(s_2dab, "ABBABBABABBABBABABBABABBABBABABBAB");
  } else if( protein_length == 38 ){
    strcpy(s_2dab, "AAAABABABABABAABAABBAAABBABAABBBABABAB");
  } else if( protein_length == 55 ){
    strcpy(s_2dab, "BABABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBAB");
  } else if( protein_length == 64 ){
    strcpy(s_2dab, "ABBABAABBABABBBAABBABABBBABBABABBABABBABABABAABABBAABBABBBAAABAB");
  } else if( protein_length == 98 ){
    strcpy(s_2dab, "AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA");
  } else if( protein_length == 120 ){
    strcpy(s_2dab, "ABBABBAABABABAABBAAAABAABABBABABBAAABBBAABBBABAAABABBABBABBBBABBBBAABBBBBBBABABBAAAABBBBBBABBBBAAAABBBABABBBBAAAABBABABB");
  } else {
    std::cout << "error string size must be 13, 21, 34, 38, 55, 64, 98, and 120.\n";
    exit(-1);
  }

  checkCudaErrors(cudaMemcpyToSymbol(S_AB, (void *) s_2dab, 150 * sizeof(char)));
  checkCudaErrors(cudaMemcpyToSymbol(PL, &protein_length, sizeof(int)));
}

F3DAB::~F3DAB()
{
  /* empty */
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__global__ void computeK_3DAB_P(float * x, float * f){
  uint id_p = blockIdx.x;
  uint id_d = threadIdx.x;
  uint ndim = params.n_dim;
  int N    = PL;

  uint THETA = id_p * ndim;
  uint BETA  = id_p * ndim + (N-2);

  __shared__ float3 points[128];

  if( id_d == 0 ){
    points[0] = make_float3(0.0f, 0.0f, 0.0f);
    points[1] = make_float3(0.0f, 1.0f, 0.0f);
    points[2] = make_float3(cosf(x[THETA]), 1.0 + sinf(x[THETA]), 0.0f);

    float3 aux = points[2];
    for( uint16_t i = 3; i < N; i++ ){
      aux.x += cosf(x[THETA + i - 2]) * cosf(x[BETA + i - 3]);
      aux.y += sinf(x[THETA + i - 2]) * cosf(x[BETA + i - 3]);
      aux.z += sinf(x[BETA  + i - 3]);

      points[i] = aux;
    }
  }

  __shared__ float v1[128], v2[128];

  v1[id_d] = 0.0;
  v2[id_d] = 0.0;

  __syncthreads();

  // if( id_d == 0 ){
  //   printf("Pontos: \n");
  //   for( uint16_t i = 0; i < N; i++ ){
  //     printf("%.3f %.3f %.3f\n", points[i].x, points[i].y, points[i].z);
  //   }
  // }

  float C, n3df, _v2;
  if( id_d < (N - 2) ){
    v1[id_d] = (1.0f - cosf(x[THETA + id_d]));

    float3 P1 = points[id_d];

    _v2 = 0.0;
    for( uint16_t j = (id_d + 2); j < N; j++ ){
      if( S_AB[id_d] == 'A' && S_AB[j] == 'A' )
        C = 1.0;
      else if( S_AB[id_d] == 'B' && S_AB[j] == 'B' )
        C = 0.5;
      else
        C = -0.5;

      float3 D = P1 - points[j];

      n3df = norm3df(D.x, D.y, D.z);

      _v2 += ( 1.0 / powf(n3df, 12.0) - C / powf(n3df, 6.0) );
    }
    v2[id_d] = _v2;
  }

  __syncthreads();

  if( id_d < 64 && N > 64 ){
    v1[id_d] += v1[id_d + 64];
    v2[id_d] += v2[id_d + 64];
  }

  __syncthreads();

  if( id_d < 32 && N > 32 ){
    v1[id_d] += v1[id_d + 32];
    v2[id_d] += v2[id_d + 32];
  }

  __syncthreads();

  if( id_d < 16 && N > 16 ){
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

    f[id_p] = (v1[0] / 4.0) + (v2[0] * 4.0);

    // printf("v1: %.4lf v2: %.4lf\n", v1[0]/4, 4*v2[0]);
    // printf("Final energy value: %.8lf\n", v1[0]/4 + 4*v2[0]);
  }
}

__global__ void computeK_3DAB_S(float *x, float *f){
  uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
  uint ps = params.ps;
  uint ndim = params.n_dim;
  int N    = PL;

  if( id_p < ps ){
    uint THETA = id_p * ndim;
    uint BETA  = id_p * ndim + (N-2);

    float3 points[128];

    points[0] = make_float3(0.0f, 0.0f, 0.0f);
    points[1] = make_float3(0.0f, 1.0f, 0.0f);
    points[2] = make_float3(cosf(x[THETA + 0]), 1 + sinf(x[THETA + 0]), 0.0f);

    float3 aux = points[2];
    for( uint16_t i = 3; i < N; i++ ){
      aux.x += cosf(x[THETA + i - 2]) * cosf(x[BETA + i - 3]);
      aux.y += sinf(x[THETA + i - 2]) * cosf(x[BETA + i - 3]);
      aux.z += sinf(x[BETA + i - 3]);

      points[i] = aux;
    }

    __syncthreads();

    // printf("Pontos: \n");
    // for( uint16_t i = 0; i < N; i++ ){
    //   printf("%.3f %.3f %.3f\n", points[i].x, points[i].y, points[i].z);
    // }

    float v1 = 0.0, v2 = 0.0, C, n3df;

    for( uint16_t i = 0; i < N-2; i++ ){
      v1 += (1.0f - cosf(x[THETA + i]));

      float3 P1 = points[i];

      for( uint16_t j = i + 2; j < N; j++ ){
        if( S_AB[i] == 'A' && S_AB[j] == 'A' ){
          C = 1;
        } else if( S_AB[i] == 'B' && S_AB[j] == 'B' ){
          C = 0.5;
        } else {
          C = -0.5;
        }

        float3 D = P1 - points[j];
        n3df = norm3df(D.x, D.y, D.z);

        v2 += ( 1.0f / powf(n3df, 12.0f) - C / powf(n3df, 6.0f) );
      }
    }
    // printf("v1: %.4f v2: %.4f\n", v1/4, 4*v2);
    // printf("Final energy value: %.8lf\n", v1/4 + 4*v2);
    f[id_p] = (v1 / 4.0) + (4.0 * v2);
  }
}

void F3DAB::compute(float * x, float * f){
  computeK_3DAB_P<<< NB, NT >>>(x, f);
  checkCudaErrors(cudaGetLastError());
}
