#include "Benchmarks.cuh"

Benchmarks::Benchmarks()
{
  min = -100.0;
  max = +100.0;
  n_dim = 100;
  protein_length = 1;
}

Benchmarks::~Benchmarks()
{
  /* empty */
}

float Benchmarks::getMin(){
  return min;
}

float Benchmarks::getMax(){
  return max;
}

uint Benchmarks::getID(){
  return ID;
}

void Benchmarks::setMin( float _min ){
  min = _min;
}

void Benchmarks::setMax( float _max ){
  max = _max;
}

void Benchmarks::setThreads( uint _n){
  NT.x = _n;
}

void Benchmarks::setBlocks( uint _n ){
  NB.x = _n;
}

uint Benchmarks::getThreads(){
  return NT.x;
}

uint Benchmarks::getBlocks(){
  return NB.x;
}
