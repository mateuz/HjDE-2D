#include "utils.hpp"

double stime(){
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  double mlsec = 1000.0 * ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
  return mlsec/1000.0;
}

void show_params( uint n_runs, uint NP, uint n_evals, uint D, uint PL, std::string FuncObj ){
  int NBA = (NP%32)? (NP/32)+1 : NP/32;
  int NTB = 32 * ceil((double) D / 32.0);

  printf(" | Number of Executions:                    %d\n", n_runs);
  printf(" | Population Size:                         %d\n", NP);
  printf(" | Protein Length:                          %d\n", PL);
  printf(" | Number of Dimensions:                    %d\n", D);
  printf(" | Number of Function Evaluations:          %d\n", n_evals);
  printf(" | Optimization Function:                   %s\n", FuncObj.c_str());
  printf(" +==============================================================+ \n");
  printf(" | Structure (A)\n");
  printf(" | \t Number of Threads                        %d\n", 32);
  printf(" | \t Number of Blocks                         %d\n", NBA);
  printf(" | Structure (B)\n");
  printf(" | \t Number of Threads                        %d\n", NTB);
  printf(" | \t Number of Blocks                         %d\n", NP);
}

std::string toString(uint id){
  switch( id ){
    case 1001:
      return "2D-AB";
    case 1002:
      return "3D-AB";
    default:
      return "Unknown";
  }
}
