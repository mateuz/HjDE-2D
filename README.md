# A GPU-Based Memetic Algorithm for the 2D Protein Structure Prediction

###### Here goes the algorithm description. 

***
##### Requirements

- ##### [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested with 9.2)

- ##### GPU Compute Capability (tested with versions 5.2 and 6.1)

- ##### [Boost C++ Libraries - Program Options](https://www.boost.org/) (tested with 1.58.0)

##### Compile

```sh
$ cd repo
$ make
```

##### Parameters Setting

```
$ "runs, r"      - Number of Executions
$ "pop_size, p"  - Population Size
$ "dim, d"       - Number of Dimensions {13, 21, 34, 38, 55, 64, 98, 120}
$ "func_obj, o"  - Function to Optimize {1001}
$ "max_eval, e"  - Number of Function Evaluations
$ "help, h"      - Show this help
```

##### Proteins Tested and Results

- Empty list for a while

##### Execute

```sh
$ cd repo
$ ./demo <parameter setting> or make run (with default parameters)
```

##### Clean up

```sh
$ make clean
```

##### TODO

- Empty list for a while

***

[1] J. Brest, V. Zumer and M. S. Maucec, "Self-Adaptive Differential Evolution Algorithm in Constrained Real-Parameter Optimization," 2006 IEEE International Conference on Evolutionary Computation, Vancouver, BC, 2006, pp. 215-222. doi: 10.1109/CEC.2006.1688311, [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1688311&isnumber=35623)

[2] [CUDA is a parallel computing platform and programming model developed by NVIDIA for GPGPU](https://developer.nvidia.com/cuda-zone)
