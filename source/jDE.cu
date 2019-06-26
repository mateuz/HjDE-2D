#include "jDE.cuh"

jDE::jDE( uint _s, uint _ndim, float _x_min, float _x_max ):
  NP(_s),
  n_dim(_ndim),
  x_min(_x_min),
  x_max(_x_max)
{
  checkCudaErrors(cudaMalloc((void **)&F,  NP * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&CR, NP * sizeof(float)));
  thrust::fill(thrust::device, F , F  + NP, 0.50);
  thrust::fill(thrust::device, CR, CR + NP, 0.90);

  checkCudaErrors(cudaMalloc((void **)&T_F,  NP * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&T_CR, NP * sizeof(float)));
  thrust::fill(thrust::device, T_F , T_F  + NP, 0.50);
  thrust::fill(thrust::device, T_CR, T_CR + NP, 0.90);

  Configuration conf;
  conf.x_min = x_min;
  conf.x_max = x_max;
  conf.ps = NP;
  conf.n_dim = n_dim;

  checkCudaErrors(cudaMemcpyToSymbol(params, &conf, sizeof(Configuration)));
  checkCudaErrors(cudaMalloc((void **)&rseq, NP * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&fseq, 3 * NP * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_states, NP * sizeof(curandStateXORWOW_t)));
  thrust::sequence(thrust::device, rseq, rseq + NP);

  NT_A.x = 32;
  NB_A.x = (NP%32)? (NP/32)+1 : NP/32;

  NT_B.x = 32 * ceil((double) n_dim / 32.0);
  NB_B.x = NP;

  std::random_device rd;
  unsigned int seed = rd();
  setup_kernel<<<NT_A, NB_A>>>(d_states, seed);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMalloc((void **)&d_states2, NP * n_dim * sizeof(curandStateXORWOW_t)));
  sk2<<<NB_B, NT_B>>>(d_states2, seed);
  checkCudaErrors(cudaGetLastError());
}

jDE::~jDE()
{
  checkCudaErrors(cudaFree(F));
  checkCudaErrors(cudaFree(CR));
  checkCudaErrors(cudaFree(T_F));
  checkCudaErrors(cudaFree(T_CR));
  checkCudaErrors(cudaFree(rseq));
  checkCudaErrors(cudaFree(fseq));
  checkCudaErrors(cudaFree(d_states));
  checkCudaErrors(cudaFree(d_states2));
}

void jDE::reset(){
  thrust::fill(thrust::device, F , F  + NP, 0.50);
  thrust::fill(thrust::device, CR, CR + NP, 0.90);

  thrust::fill(thrust::device, T_F , T_F  + NP, 0.50);
  thrust::fill(thrust::device, T_CR, T_CR + NP, 0.90);
}

void jDE::update(){
  updateK<<<NB_A, NT_A>>>(d_states, F, CR, T_F, T_CR);
  checkCudaErrors(cudaGetLastError());
}


/*
 * fog == fitness of the old offspring
 * fng == fitness of the new offspring
 */
void jDE::run(float * og, float * ng){
  rand_DE<<<NB_B, NT_B>>>(d_states2, og, ng, T_F, T_CR, fseq);
  checkCudaErrors(cudaGetLastError());
}

void jDE::run_b(float * og, float * ng, float * bg, float * fog, float * fng, uint b_id){
  best_DE<<<NB_B, NT_B>>>(og, ng, bg, fog, fng, b_id);
  checkCudaErrors(cudaGetLastError());
}

void jDE::index_gen(){
  iGen<<<NB_A, NT_A>>>(d_states, rseq, fseq);
  checkCudaErrors(cudaGetLastError());
}

void jDE::selection(float * og, float * ng, float * fog, float * fng){
  selectionK<<<NB_A, NT_A>>>(og, ng, fog, fng);
  checkCudaErrors(cudaGetLastError());
}

void jDE::crowding_selection(
  float * og, float * ng,
  float * fog, float * fng,
  float * res
){
  float * iter;
  int position;
  thrust::device_ptr<float> d_fog = thrust::device_pointer_cast(fog);
  thrust::device_ptr<float> d_fng = thrust::device_pointer_cast(fng);

  thrust::device_ptr<float> d_f = thrust::device_pointer_cast(F);
  thrust::device_ptr<float> d_cr = thrust::device_pointer_cast(CR);

  thrust::device_ptr<float> d_tf = thrust::device_pointer_cast(T_F);
  thrust::device_ptr<float> d_tcr = thrust::device_pointer_cast(T_CR);
  // thrust::device_ptr<float> d_og = thrust::device_pointer_cast(og);
  // thrust::device_ptr<float> d_ng = thrust::device_pointer_cast(ng);

  // printf("NP: %d\n", NP);
  for( uint p = 0; p < NP; p++ ){
    crowding<<<NB_A, NT_A>>>(ng, og, p, res);
    checkCudaErrors(cudaGetLastError());
    // thrust::device_vector<float> d_res( res, res+NP);
    // thrust::host_vector<float> h_res = d_res;
    // for( int i = 0 ; i < NP; i++ ){
    //   printf("%.1f ", h_res[i]);
    // }
    // printf("\n");
    iter = thrust::min_element(thrust::device, res, res + NP);
    position = iter - res;
    //
    // printf("[%d] The minimum distance element is: %i %.3f\n", p+1, position, h_res[position]);
    // printf("d_fng[%d] %.4f <= d_fog[%d] %.4f\n", p, (float)d_fng[p], position, (float)d_fog[position]);
    if( (float)d_fng[p] <= (float)d_fog[position] ){
      // printf("Antes: %.3f e \n", (float)d_fog[position]);
      // for( int i = 0; i < n_dim; i++ ){
      //   printf("%.1f ", (float)d_og[position * n_dim + i]);
      // }
      // printf("\n");

      thrust::copy_n(thrust::device,
        ng + (p * n_dim),       //source
        n_dim,                  //num elements to copy
        og + (position * n_dim) //destination
      );

      thrust::copy_n(thrust::device,
        fng + p,       //source fitness
        1,             //copy just one value
        fog + position //destination fitness update
      );

      // update F and CR;
      d_f[position]  = d_tf[p];
      d_cr[position] = d_tcr[p];

      // printf("Agora: %.3f e \n", (float)d_fog[position]);
      // for( int i = 0; i < n_dim; i++ ){
      //   printf("%.1f ", (float)d_og[position * n_dim + i]);
      // }
      // printf("\n");
      //
      // printf("E deve ser: \n");
      // for( int i = 0; i < n_dim; i++ ){
      //   printf("%.1f ", (float)d_ng[p * n_dim + i]);
      // }
      // printf("\n");

    }
  }
  // scanf("%d", &position);
}

/*
 * Update F and CR values accordly with jDE algorithm.
 *
 * F_Lower, F_Upper and T are constant variables declared
 * on constants header
 */
__global__ void updateK(curandState * g_state, float * d_F, float * d_CR, float * d_TF, float * d_TCR) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  uint ps = params.ps;

  if( index < ps ){
    curandState localState;
    localState = g_state[index];

    //(0, 1]
    float r1, r2, r3, r4;
    r1 = curand_uniform(&localState);
    r2 = curand_uniform(&localState);
    r3 = curand_uniform(&localState);
    r4 = curand_uniform(&localState);

    if (r2 < T){
      d_TF[index] = F_Lower + (r1 * F_Upper);
    } else {
      d_TF[index] = d_F[index];
    }

    if (r4 < T){
      d_TCR[index] = r3;
    } else {
      d_TCR[index] = d_CR[index];
    }

    g_state[index] = localState;
  }
}

/*
 * Performs the selection step
 * In this case, each thread is a individual
 * og -> Old genes, the previous generation offspring
 * ng -> New genes, the new generation offsprings
 * fog -> fitness of the old offspring
 * fng -> fitness of the new offspring
 */
__global__ void selectionK(float * og, float * ng, float * fog, float * fng){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;
  uint ps = params.ps;

  if( index < ps ){
    uint ndim = params.n_dim;
    if( fng[index] <= fog[index] ){
      memcpy(og + (ndim * index), ng + (ndim * index), ndim * sizeof(float));
      fog[index]  = fng[index];
   }
  }
}

/*
 * Performs the DE/rand/1/bin operation
 * 1 thread == 1 individual
 * rng == global random state
 * fog == fitness of the old offspring
 * fng == fitness of the new offspring
 * F == mutation factor vector
 * CR == crossover probability vector
 */
__global__ void DE(curandState * rng, float * og, float * ng, float * F, float * CR, uint * fseq){
  uint i, index, ps, n_dim;
  index = threadIdx.x + blockDim.x * blockIdx.x;
  ps = params.ps;

  if(index < ps){
    uint n1, n2, n3, p1, p2, p3, p4;
    n_dim = params.n_dim;

    float mF  = F[index];
    float mCR = CR[index];

    curandState random = rng[index];

    n1 = fseq[index];
    n2 = fseq[index + ps];
    n3 = fseq[index + ps + ps];

    //do n1 = curand(&random)%ps; while (n1 == index);
    //do n2 = curand(&random)%ps; while (n2 == index || n2 == n1 );
    //do n3 = curand(&random)%ps; while (n3 == index || n3 == n1 || n3 == n2);

    p1 = index * n_dim;
    p2 = n3 * n_dim;
    p3 = n2 * n_dim;
    p4 = n1 * n_dim;
    //printf("[%u] %u %u %u => %u %u %u %u\n", index, n1, n2, n3, p4, p3, p2, p1);
    for( i = 0; i < n_dim; i++ ){
      if( curand_uniform(&random) <= mCR || (i == n_dim - 1) ){
        /* Get three mutually different indexs */
        ng[p1 + i] = og[p2 + i] + mF * (og[p3 + i] - og[p4 + i]);

        /* Check bounds */
        ng[p1 + i] = max(params.x_min, ng[p1 + i]);
        ng[p1 + i] = min(params.x_max, ng[p1 + i]);
      } else {
        ng[p1 + i] = og[p1 + i];
      }
    }
    rng[index] = random;
  }
}

__global__ void rand_DE(curandState *rng, float * og, float * ng, float * F, float * CR, uint * fseq){
  uint id_d, id_p, ps, n_dim;

  //id_g = threadIdx.x + blockDim.x * blockIdx.x;

  id_d = blockIdx.x;
	id_p = threadIdx.x;

  n_dim = params.n_dim;
  ps = params.ps;

  __syncthreads();

  if( id_p < n_dim ){
    curandState random = rng[ id_d * id_p ];

    __shared__ uint n1, n2, n3, p1, p2, p3, p4, rnbr;
    __shared__ float mF, mCR;

    if( id_p == 0 ){
      n1 = fseq[id_d];
      n2 = fseq[id_d + ps];
      n3 = fseq[id_d + ps + ps];

      mF  = F[id_d];
      mCR = CR[id_d];

      p1 = id_d * n_dim;
      p2 = n3 * n_dim;
      p3 = n2 * n_dim;
      p4 = n1 * n_dim;

      rnbr = curand(&random) % n_dim;
    }

    __syncthreads();

    if( curand_uniform(&random) <= mCR || (id_p == rnbr) ){
      ng[p1 + id_p] = og[p2 + id_p] + mF * (og[p3 + id_p] - og[p4 + id_p]);

      // ng[p1 + id_p] = max(params.x_min, ng[p1 + id_p]);
      // ng[p1 + id_p] = min(params.x_max, ng[p1 + id_p]);
      if( ng[p1 + id_p] <= params.x_min ){
        ng[p1 + id_p] += 2.0 * params.x_max;
      } else if( ng[p1 + id_p] > params.x_max ){
        ng[p1 + id_p] += 2.0 * params.x_min;
      }
    } else {
      ng[p1 + id_p] = og[p1 + id_p];
    }

    rng[id_d * id_p ] = random;
  }
}

__global__ void best_DE(float * og, float * ng, float * bnew, float * fog, float * fng, uint pbest){
  uint id_d, id_p, n_dim;

  //id_g = threadIdx.x + blockDim.x * blockIdx.x;

  id_d  = blockIdx.x;
	id_p  = threadIdx.x;

  n_dim = params.n_dim;

  if( id_p < n_dim ){
    __shared__ uint p1;
    __shared__ uint pb;

    __shared__ float _FA;
    __shared__ float _FB;

    __syncthreads();

    if( id_p == 0 ){
      p1 = id_d * n_dim;
      pb = pbest * n_dim;

      _FA = fog[id_d];
      _FB = fng[id_d];
    }

    __syncthreads();

    // if(id_p == 0 && id_d == 0){
    //   for(int i = 0; i < n_dim; i++){
    //     printf("teste[%d] = %.3f;\n", i, og[pb + i]);
    //   }
    // }

    if( _FB <= _FA ){
      bnew[p1 + id_p] = og[pb + id_p] + 0.5 * (ng[p1 + id_p] - og[p1 + id_p]);

      //check bounds
      if( bnew[p1 + id_p] <= params.x_min ){
        bnew[p1 + id_p] += 2.0 * params.x_max;
      } else if( bnew[p1 + id_p] > params.x_max ){
        bnew[p1 + id_p] += 2.0 * params.x_min;
      }
    } else {
      bnew[p1 + id_p] = ng[p1 + id_p];
    }
  }
}

/*
 * Performs the crowding operation
 * One thread per individual
 *
 * The kernel calculate the projection of a vector (A)
 * in a set of vectors (B)
 *
 * @params:
 * float * A: fixed vector to compare
 * float * B: array to compare
 * uint pid: the fixed index to compare
 * float * res: stores the distance;
 */
__global__ void crowding(float * A, float * B, uint p_id, float * res){
  uint t_id = threadIdx.x + blockDim.x * blockIdx.x;

  uint N  = params.n_dim;
  uint PS = params.ps;

  if( t_id < PS ){
    float S = 0.0; //stores the sum
    float D = 0.0; //stores the distance

    for( uint i = 0; i < N; i++ ){
      D = A[p_id * N + i] - B[t_id * N + i];
      S += D * D;
    }

    res[t_id] = S;
  }

}

/*
 * Generate 3 different indexs to DE/rand/1/bin.
 * @TODO:
 *  + rseq on constant memory;
 */
__global__ void iGen(curandState * g_state, uint * rseq, uint * fseq){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;

  uint ps = params.ps;
  if( index < ps ){
    curandState s = g_state[index];

    uint n1, n2, n3;

    n1 = curand(&s) % ps;
    if( rseq[n1] == index )
      n1 = (n1 + 1) % ps;

    n2 = ( curand(&s) % ((int)ps/3) ) + 1;
    if( rseq[(n1 + n2) % ps] == index )
      n2 = (n2 + 1) % ps;

    n3 = ( curand(&s) % ((int)ps/3) ) + 1;
    if( rseq[(n1 + n2 + n3) % ps] == index )
      n3 = (n3 + 1 ) % ps;

    fseq[index] = rseq[n1];
    fseq[index+ps] = rseq[(n1+n2)%ps];
    fseq[index+ps+ps] = rseq[(n1+n2+n3)%ps];

    g_state[index] = s;
    //printf("[%-3d] %-3d | %-3d | %-3d\n", index, rseq[n1], rseq[(n1+n2)%ps], rseq[(n1+n2+n3)%ps]);
  }
}

/* Each thread gets same seed, a different sequence number, no offset */
__global__ void setup_kernel(curandState * random, uint seed){
  uint index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index < params.ps)
    curand_init(seed, index, 0, &random[index]);
}

/*
 *
 * Setup kernel version 2
 *
 */
__global__ void sk2(curandState * random, uint seed){
  uint index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index < params.ps * params.n_dim)
    curand_init(seed, index, 0, &random[index]);
}
