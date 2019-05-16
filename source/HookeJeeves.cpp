#include "HookeJeeves.hpp"

#define PI 3.14159265

// Parameters received:
//   - uint: number of Dimensions
//   - double: rho
//   - double: epsilon
//   - Benchmarks: function to be optimized

HookeJeeves::HookeJeeves(uint _nd, double _rho, double _e){
  nvars   = _nd;
  rho     = _rho;
  epsilon = _e;

  // printf(" | Number of Dimensions:        %d\n", nvars);
  // printf(" | Rho:                         %.3lf\n", rho);
  // printf(" | Epsilon                      %.3lf\n", epsilon);

  startpt = new float[nvars];
  delta   = new float[nvars];

  newx    = new float[nvars];
  xbef    = new float[nvars];
  z       = new float[nvars];

  amino_pos = new amino[nvars];
  //(amino *) malloc (nvars * sizeof (amino));

  if( nvars == 13 ){
    AB_SQ = "ABBABBABABBAB";
  } else if( nvars == 21 ){
    AB_SQ = "BABABBABABBABBABABBAB";
  } else if( nvars == 34 ){
    AB_SQ = "ABBABBABABBABBABABBABABBABBABABBAB";
  } else if( nvars == 38 ){
    AB_SQ = "AAAABABABABABAABAABBAAABBABAABBBABABAB";
  } else if( nvars == 55 ){
    AB_SQ = "BABABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBAB";
  } else if( nvars == 64 ){
    AB_SQ = "ABBABAABBABABBBAABBABABBBABBABABBABABBABABABAABABBAABBABBBAAABAB";
  } else if( nvars == 98 ){
    AB_SQ = "AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA";
  } else if( nvars == 120 ){
    AB_SQ = "ABBABBAABABABAABBAAAABAABABBABABBAAABBBAABBBABAAABABBABBABBBBABBBBAABBBBBBBABABBAAAABBBBBBABBBBAAAABBBABABBBBAAAABBABABB";
  } else {
    std::cout << "Error, AB string string sequence only defined to 13, 21, 34, 38, 55, 64, 98, and 120.\n";
    exit(-1);
  }

  memset(delta, 0, sizeof(float) * nvars);
}

HookeJeeves::~HookeJeeves(){
  delete [] startpt;
  delete [] delta;
  delete [] amino_pos;
  delete [] newx;
  delete [] xbef;
}

float HookeJeeves::evaluate(float * individual){
  double a_ab,b_ab,c_ab,d_ab,v1,v2;

  unsigned int i = 0, j = 0;

  amino_pos[0].x = 0;
  amino_pos[0].y = 0;
  amino_pos[1].x = 1;
  amino_pos[1].y = 0;

  for (i = 1; i < (nvars - 1); i++){
    a_ab = amino_pos[i].x-amino_pos[i-1].x;
    b_ab = amino_pos[i].y-amino_pos[i-1].y;
    amino_pos[i+1].x = amino_pos[i].x+a_ab*cos(individual[i-1])-b_ab*sin(individual[i-1]);
    amino_pos[i+1].y = amino_pos[i].y+b_ab*cos(individual[i-1])+a_ab*sin(individual[i-1]);
  }

  v1 = 0;
  v2 = 0;
  for (i = 0; (i < (nvars-2)); i++) {
    v1 += (1.0 - cos(individual[i]) ) / 4.0;
    for (j = (i+2); (j < nvars); j++) {
      if (AB_SQ[i] == 'A' && AB_SQ[j] == 'A') //AA bond
        c_ab = 1;
      else if (AB_SQ[i] == 'B' && AB_SQ[j] == 'B') //BB bond
        c_ab = 0.5;
      else
        c_ab = -0.5; //AB or BA bond

      d_ab = sqrt(((amino_pos[i].x-amino_pos[j].x)*(amino_pos[i].x-amino_pos[j].x))+((amino_pos[i].y-amino_pos[j].y)*(amino_pos[i].y-amino_pos[j].y))); //Distance for Lennard-Jones
      v2 += 4.0 * ( 1 / pow(d_ab, 12) - c_ab / pow(d_ab, 6) );
    }
  }
  return(v1 + v2);
}

float HookeJeeves::best_nearby(float * point, float prevbest, uint * eval){
  float minf, ftmp;

  // save point on z
  memcpy(z, point, sizeof(float) * nvars);

  minf = prevbest;

  for( uint i = 0; i < nvars; i++ ){
    z[i] = point[i] + delta[i];

    // check bounds
    if( z[i] <= -PI ){
      z[i] += 2.0 * PI;
    } else if(z[i] > +PI ){
      z[i] += 2.0 * -PI;
    }

    ftmp = evaluate(z);
    (*eval)++;

    if( ftmp < minf ){
      minf = ftmp;
    } else {
      delta[i] = - delta[i];
      z[i] = point[i] + delta[i];

      // check bounds
      if( z[i] <= -PI ){
        z[i] += 2.0 * PI;
      } else if(z[i] > +PI){
        z[i] += 2.0 * -PI;
      }

      ftmp = evaluate(z);
      (*eval)++;

      if( ftmp < minf )
        minf = ftmp;
      else
        z[i] = point[i];
    }
  }
  memcpy(point, z, sizeof(float) * nvars);

  return minf;
}

float HookeJeeves::optimize(const uint n_evals, float * _startpt){
  bool keep_on;

  memcpy(newx, _startpt, sizeof(float) * nvars);
  memcpy(xbef, _startpt, sizeof(float) * nvars);

  uint it;
  for( it = 0; it < nvars; it++ ){
    delta[it] = fabs(_startpt[it] * rho);
    if( delta[it] == 0.0 )
      delta[it] = rho;
  }

  float fbef;
  float fnew;
  float tmp;

  //fbef = B->compute(newx, 0);
  fbef = evaluate(newx);
  // printf("HJ starts the process with %.10f and ", fbef);

  fnew = fbef;

  double step_length = rho;

  it = 0;
  while( it < n_evals && step_length > epsilon ){
    // it++;
    // printf("\nAfter %5d fun evals, f(x) = %.4lf\n", it, fbef);

    memcpy(newx, xbef, sizeof(float) * nvars);

    fnew = best_nearby(newx, fbef, &it);

    // if we made some improvements, pursue that direction
    keep_on = true;
    // printf("[1] %.10lf\n", fnew);
    while( (fnew < fbef) && (keep_on == true) ){
      for( uint i = 0; i < nvars; i++ ){

        // firstly, arrange the sign of delta[]
        if( newx[i] <= xbef[i] )
          delta[i] = -fabs(delta[i]);
        else
          delta[i] = fabs(delta[i]);

        // now, move further in this direction
        tmp     = xbef[i];
        xbef[i] = newx[i];
        newx[i] = newx[i] + newx[i] - tmp;

        // check bounds
        if( newx[i] <= -PI ){
          newx[i] += 2.0 * PI;
        } else if(newx[i] > +PI ){
          newx[i] += 2.0 * -PI;
        }
      }
      fbef = fnew;

      fnew = best_nearby(newx, fbef, &it);

      if( it > n_evals ) break;

      // if the further (optimistic) move was bad
      if( fnew >= fbef ) break;

      keep_on = false;
      for( uint i = 0; i < nvars; i++ ){
        keep_on = true;
        if( fabs(newx[i] - xbef[i]) > (0.5 * fabs(delta[i])) )
          break;
        else
          keep_on = false;
      }
    }
    // printf("[2] %.10lf\n", fnew);
    if( (step_length >= epsilon) and (fnew >= fbef) ){
      step_length *= rho;
      for( uint i = 0; i < nvars; i++ )
        delta[i] *= rho;
    }
  }

  // copy the improved result to startpt
  for( uint i = 0; i < nvars; i++ )
    _startpt[i] = xbef[i];

  // printf("exits with %.10f\n", fbef);
  return fbef;
}
