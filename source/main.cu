/* CUDA includes */
#include "helper.cuh"

/* Local includes */
#include "utils.hpp"
#include "Benchmarks.cuh"
#include "jDE.cuh"
#include "2D_AB.cuh"
#include "HookeJeeves.hpp"

#define SAVE 1

int main(int argc, char * argv[]){
  srand(time(NULL));
  uint n_runs, NP, n_evals, PL, f_id = 1001;

  try {
    po::options_description config("Opções");
    config.add_options()
      ("runs,r"    , po::value<uint>(&n_runs)->default_value(1)    , "Number of Executions"          )
      ("pop_size,p", po::value<uint>(&NP)->default_value(20)       , "Population Size"               )
      ("protein_lenght,d", po::value<uint>(&PL)->default_value(13) , "Protein Length"                )
      ("max_eval,e", po::value<uint>(&n_evals)->default_value(10e5), "Number of Function Evaluations")
      ("help,h", "Show help");

    po::options_description cmdline_options;
    cmdline_options.add(config);
    po::variables_map vm;
    store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    notify(vm);
    if( vm.count("help") ){
      std::cout << cmdline_options << "\n";
      return 0;
    }
  } catch(std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }

  uint n_dim = PL;

  printf(" +==============================================================+ \n");
  printf(" |                      EXECUTION PARAMETERS                    | \n");
  printf(" +==============================================================+ \n");
  show_params(n_runs, NP, n_evals, n_dim, PL, toString(f_id));
  printf(" +==============================================================+ \n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  thrust::device_vector<float> d_og(n_dim * NP);
  thrust::device_vector<float> d_ng(n_dim * NP);
  thrust::device_vector<float> d_bg(n_dim * NP);
  thrust::device_vector<float> d_fog(NP, 0.0);
  thrust::device_vector<float> d_fng(NP, 0.0);
  thrust::device_vector<float> d_fbg(NP, 0.0);
  thrust::device_vector<float> d_res(NP, 0.0);

  thrust::host_vector<float> h_og(n_dim * NP);
  thrust::host_vector<float> h_ng(n_dim * NP);
  thrust::host_vector<float> h_bg(n_dim * NP);
  thrust::host_vector<float> h_fog(NP);
  thrust::host_vector<float> h_fng(NP);
  thrust::host_vector<float> h_fbg(NP);

  float * p_og  = thrust::raw_pointer_cast(d_og.data());
  float * p_ng  = thrust::raw_pointer_cast(d_ng.data());
  float * p_bg  = thrust::raw_pointer_cast(d_bg.data());
  float * p_fog = thrust::raw_pointer_cast(d_fog.data());
  float * p_fng = thrust::raw_pointer_cast(d_fng.data());
  float * p_fbg = thrust::raw_pointer_cast(d_fbg.data());
  float * p_res = thrust::raw_pointer_cast(d_res.data());

  thrust::device_vector<float>::iterator it;
  int b_id;

  Benchmarks * B = NULL;
  if( f_id == 1001 )
    B = new F2DAB(n_dim, NP);

  if( B == NULL ){
    printf("Unknown function! Exiting...\n");
    exit(EXIT_FAILURE);
  }

  float x_min = B->getMin();
  float x_max = B->getMax();

  float time  = 0.00;
  jDE * jde = new jDE(NP, n_dim, x_min, x_max);
  HookeJeeves * hj  = new HookeJeeves(n_dim, 0.9, 1.0e-30);
  double hjres = 0;

  std::vector< std::pair<double, float> > stats;

  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> random_i(0, NP-1);//[0, NP-1]

  // stores the iteration, best, mean, worst, and Isd;
  std::vector< std::tuple<uint, float, float, float, float > > data;

  for( uint run = 1; run <= n_runs; run++ ){
    // Randomly initiate the population

    // For practical use
    // random_device is generally only used to seed
    // a PRNG such as mt19937
    std::random_device rd;

    thrust::counting_iterator<uint> isb(0);
    thrust::transform(isb, isb + (n_dim * NP), d_og.begin(), prg(x_min, x_max, rd()));

    /* Starts a Run */

    //warm-up
    B->compute(p_og, p_fog);

    // get the best index
    it   = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
    b_id = thrust::distance(d_fog.begin(), it);

    int g = 0;
    float gb_e = static_cast<float>(*it);

    cudaEventRecord(start);
    for( uint evals = 0; evals < n_evals; ){
      if( SAVE == 1 ){
        // copy data to host
        h_og = d_og;
        h_fog = d_fog;

        // look for the best solution
        thrust::host_vector<float>::iterator it = thrust::min_element(thrust::host, h_fog.begin(), h_fog.end());
        float best = static_cast<float>(*it);

        if( gb_e < best ) {
          best = gb_e;
        }

        // look for the average
        float average = 0.0;
        // average = (thrust::reduce(thrust::host, h_fog.begin(), h_fog.end())) / (NP*1.0);

        // look for the worst solution
        // it = thrust::max_element(thrust::host, h_fog.begin(), h_fog.end());
        float worst = 0.0;
        // worst = static_cast<float>(*it);

        //Diversity calc :: moment of inertia about the centroids
        std::vector<float> ci;
        for( int d = 0; d < n_dim; d++ ){
          float _ci = 0.0;
          for( int p = 0; p < NP; p++ ){
            _ci += h_og[(p * n_dim) + d] / (float) NP;
          }
          ci.push_back(_ci);
        }

        float Isd = 0.0, _t1, _t2;
        for( int d = 0; d < n_dim; d++ ){
          _t1 = 0.0;
          for( int p = 0; p < NP; p++ ){
            _t2 = h_og[(p * n_dim) + d] - ci[d];
            _t1 += (_t2 * _t2);
          }
          Isd += sqrt(_t1 / (NP - 1));
        }
        Isd /= (double) n_dim;
        std::cout << best << " :: " << average << " :: " << worst << " :: " << Isd << std::endl;
        if( best < 0.0 ){
          data.push_back( std::make_tuple (g, best, average, worst, Isd) );
        }
      }

      jde->index_gen();
      jde->run(p_og, p_ng);
      B->compute(p_ng, p_fng);
      evals += NP;

      // get the best index
      it   = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
      b_id = thrust::distance(d_fog.begin(), it);
      gb_e = static_cast<float>(*it);

      jde->run_b(p_og, p_ng, p_bg, p_fog, p_fng, b_id);
      B->compute(p_bg, p_fbg);
      evals += NP;

      //selection between trial and best variations
      jde->selection(p_ng, p_bg, p_fng, p_fbg);

      //crowding between old generation and new trial vectors
      jde->crowding_selection(p_og, p_ng, p_fog, p_fng, p_res);

      jde->update();

      if( g%1000 == 0 && g != 0 ){
        int b_idx = random_i(rng);

        thrust::host_vector<double> H(n_dim);

        // copy from device to host
        for( int d = 0; d < n_dim; d++ )
          H[d] = static_cast<double>(d_og[(b_idx * n_dim) + d]);

        // apply hooke-jeeves to the solution vector selected
        d_fog[b_idx] = static_cast<float>(hj->optimize(10000, H.data()));

        // copy from host to device
        for( int d = 0; d < n_dim; d++ )
          d_og[(b_idx * n_dim) + d] = static_cast<float>(H[d]);
      }
      g++;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    /* End a Run */

    float * iter = thrust::min_element(thrust::device, p_fog, p_fog + NP);
    int position = iter - p_fog;

    thrust::host_vector<double> H(n_dim);
    for( int nd = 0; nd < n_dim; nd++ ){
      H[nd] = static_cast<double>(d_og[(position * n_dim) + nd]);
    }
    double tini, tend;

    tini = stime();
    hjres = hj->optimize(1000000, H.data());
    tend = stime();

    printf(" | Conformation: \n | ");
    for( int nd = 0; nd < n_dim; nd++ ){
      printf("%.30lf ", H[nd]);
    }
    printf("\n");

    printf(" | Execution: %-2d Overall Best: %+.4f -> %+.4lf GPU Time (s): %.8f and HJ Time (s): %.8f\n", run, static_cast<float>(*it), hjres, time/1000.0, tend-tini);

    data.push_back( std::make_tuple (g, hjres, 0.0, 0.0, 0.0) );

    stats.push_back(std::make_pair(hjres, time));

    jde->reset();
  }

  /* Statistics of the Runs */
  float FO_mean = 0.0f, FO_std = 0.0f;
  float T_mean  = 0.0f, T_std  = 0.0f;
  for( auto it = stats.begin(); it != stats.end(); it++){
    FO_mean += it->first;
    T_mean  += it->second;
  }
  FO_mean /= n_runs;
  T_mean  /= n_runs;
  for( auto it = stats.begin(); it != stats.end(); it++){
    FO_std += (( it->first - FO_mean )*( it->first  - FO_mean ));
    T_std  += (( it->second - T_mean )*( it->second - T_mean  ));
  }
  FO_std /= n_runs;
  FO_std = sqrt(FO_std);
  T_std /= n_runs;
  T_std = sqrt(T_std);
  printf(" +==============================================================+ \n");
  printf(" |                     EXPERIMENTS RESULTS                      | \n");
  printf(" +==============================================================+ \n");
  printf(" | Objective Function:\n");
  printf(" | \t mean:         %+.20E\n", FO_mean);
  printf(" | \t std:          %+.20E\n", FO_std);
  printf(" | Execution Time (ms): \n");
  printf(" | \t mean:         %+.3lf\n", T_mean);
  printf(" | \t std:          %+.3lf\n", T_std);
  printf(" +==============================================================+ \n");
  
  if( SAVE == 1 ){
    std::ofstream ofs_csv;
    std::ofstream ofs_json_convergence;
    std::ofstream ofs_json_diversity;

    ofs_csv.open("results/output.csv", std::ofstream::out);
    ofs_json_convergence.open("results/output_conv.json", std::ofstream::out);
    ofs_json_diversity.open("results/output_div.json", std::ofstream::out);

    if( not ofs_csv.is_open() )
      std::cout << "Error opening data output file" << std::endl;
    else {
      ofs_csv << "gen,global_best,global_average,global_worst,diversity\n";
      for( auto it = data.begin(); it != data.end(); it++ )
        ofs_csv << std::get<0>(*it) << "," << std::get<1>(*it) << "," << std::get<2>(*it) << "," << std::get<3>(*it) << "," << std::get<4>(*it) << "\n";

      save_json(data, ofs_json_convergence);
      save_diversity(data, ofs_json_diversity);

      ofs_csv.close();
    }
  }

  return 0;
}
