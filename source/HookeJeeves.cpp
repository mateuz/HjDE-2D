#include "HookeJeeves.hpp"

// Parameters received:
//   - uint: number of Dimensions
//   - double: rho
//   - double: epsilon
HookeJeeves::HookeJeeves(uint _nd, double _rho, double _e){
  nvars   = _nd;
  rho     = _rho;
  epsilon = _e;

  // printf(" | Number of Dimensions:        %d\n", nvars);
  // printf(" | Rho:                         %.3lf\n", rho);
  // printf(" | Epsilon                      %.3lf\n", epsilon);

  startpt = new double[nvars];
  delta   = new double[nvars];
  newx    = new double[nvars];
  xbef    = new double[nvars];
  z       = new double[nvars];

  amino_pos = new amino[nvars];
  memset(delta, 0, sizeof(double) * nvars);

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
}

HookeJeeves::~HookeJeeves(){
  delete [] startpt;
  delete [] delta;
  delete [] amino_pos;
  delete [] newx;
  delete [] xbef;
}

double HookeJeeves::evaluate(const double * gen){
  int i, j;
  double d_x, d_y, v1, v2, C, D;

  amino_pos[0].x = 0.0;
  amino_pos[0].y = 0.0;
  amino_pos[1].x = 1.0;
  amino_pos[1].y = 0.0;

  for( i = 1; i < (nvars - 1); i++ ){
    d_x = amino_pos[i].x - amino_pos[i-1].x;
    d_y = amino_pos[i].y - amino_pos[i-1].y;

    amino_pos[i+1].x = amino_pos[i].x + d_x * cos( gen[i - 1]) - d_y * sin( gen[i - 1] );
    amino_pos[i+1].y = amino_pos[i].y + d_y * cos( gen[i - 1]) + d_x * sin( gen[i - 1] );
  }

  v1 = v2 = 0.0;

  for( i = 0; i < (nvars - 2); i++ ){
    v1 += (1.0 - cos(gen[i])) / 4.0;

    for( j = (i+2); j < nvars; j++ ){

      if( AB_SQ[i] == 'A' && AB_SQ[j] == 'A' )
        C = 1;
      else if( AB_SQ[i] == 'B' && AB_SQ[j] == 'B' )
        C = 0.50;
      else
        C = -0.50;

      d_x = amino_pos[i].x - amino_pos[j].x;
      d_y = amino_pos[i].y - amino_pos[j].y;

      D = sqrt( (d_x * d_x) + (d_y * d_y) );
      v2 += 4.0 * ( 1.0/pow(D, 12.0) - C/pow(D, 6.0) );
    }

  }
  return v1 + v2;
}

double HookeJeeves::best_nearby(double * point, double prevbest, uint * eval){
  double minf, ftmp;

  // save point on z
  memcpy(z, point, sizeof(double) * nvars);

  minf = prevbest;

  for( uint i = 0; i < nvars; i++ ){
    z[i] = point[i] + delta[i];

    // check bounds
    if( z[i] <= -PI ){
      z[i] += 2.0 * PI;
    } else if(z[i] > PI ){
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
      } else if(z[i] > PI ){
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
  memcpy(point, z, sizeof(double) * nvars);

  return minf;
}

double HookeJeeves::optimize(const uint n_evals, double * _startpt){
  bool keep_on;

  memcpy(newx, _startpt, sizeof(double) * nvars);
  memcpy(xbef, _startpt, sizeof(double) * nvars);

  uint it;
  for( it = 0; it < nvars; it++ ){
    delta[it] = fabs(_startpt[it] * rho);
    if( delta[it] == 0.0 )
      delta[it] = rho;
  }

  double fbef;
  double fnew;
  double tmp;

  fbef = evaluate(newx);

  fnew = fbef;

  double step_length = rho;

  it = 0;
  while( it < n_evals && step_length > epsilon ){
    memcpy(newx, xbef, sizeof(double) * nvars);

    fnew = best_nearby(newx, fbef, &it);

    // if we made some improvements, pursue that direction
    keep_on = true;
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
        } else if(newx[i] > PI ){
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
    if( (step_length >= epsilon) and (fnew >= fbef) ){
      step_length *= rho;
      for( uint i = 0; i < nvars; i++ )
        delta[i] *= rho;
    }
  }

  // copy the improved result to startpt
  for( uint i = 0; i < nvars; i++ )
    _startpt[i] = xbef[i];

  return fbef;
}
