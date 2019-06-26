#ifndef _UTILS_H
#define _UTILS_H

/* C++ includes */

#include <iostream>
#include <chrono>
#include <functional>
#include <iomanip>
#include <random>
#include <algorithm>
#include <string>

/* C includes */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

double stime();

void show_params(uint, uint, uint, uint, uint, std::string);

std::string toString(uint);

#endif
