#ifndef _BENCHMARKS_H
#define _BENCHMARKS_H

class Benchmarks
{
protected:
  float min;
  float max;

  uint ID;
  uint n_dim;
  uint ps;
  uint protein_length;

  dim3 NT;
  dim3 NB;
public:

  Benchmarks();
  virtual ~Benchmarks();

  virtual void compute(float * x, float * fitness)
  {
    /* empty */
  };

  float getMin();
  float getMax();
  uint getID();

  void setMin( float );
  void setMax( float );

  /* GPU launch compute status */
  void setThreads( uint );
  void setBlocks( uint );

  uint getThreads();
  uint getBlocks();
};

#endif
