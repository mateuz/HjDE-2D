#include "utils.hpp"

double stime(){
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  double mlsec = 1000.0 * ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
  return mlsec/1000.0;
}

void show_params(
	uint n_runs,
	uint NP,
	uint n_evals,
	uint n_dim,
	std::string FuncObj
){
	printf(" | Number of Executions:                    %d\n", n_runs);
	printf(" | Population Size:                         %d\n", NP);
	printf(" | Number of Dimensions:                    %d\n", n_dim);
	printf(" | Number of Function Evaluations:          %d\n", n_evals);
	printf(" | Optimization Function:                   %s\n", FuncObj.c_str());
	printf(" +==============================================================+ \n");
	printf(" | Number of Threads                        %d\n", 32);
	printf(" | Number of Blocks                         %d\n", (NP%32)? (NP/32)+1 : NP/32);
}

std::string toString(uint id){
  switch( id ){
    case 1001:
      return "2D-AB";
    default:
      return "Unknown";
  }
}
