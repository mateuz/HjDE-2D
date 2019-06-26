# A GPU-Based Hybrid jDE Algorithm Applied to the 2D-AB Protein Structure Prediction

###### Protein Structure Prediction (PSP) problem is an open problem in bioinformatics and, as the problem scales, complexity and processing time increases. In this way, robust methods and massively parallel architectures are required. This repository provide a GPU-based hybrid algorithm, named cuHjDE, to handle the 2D-AB off-lattice PSP problem. The cuHjDE is composed of the jDE algorithm and the Hooke-Jeeves local search algorithm. An important feature present in the cuHjDE algorithm is the use of a crowding mechanism to avoid premature convergence promoting diversification in the search space. The proposed algorithm is compared with four state-of-the-art algorithms in both artificial and real proteins. Also, the impact of using GPU is analyzed. Experimental results point out that the proposed approach is competitive in artificial sequences and achieved new best results for all real sequence proteins.


***
##### Requirements

- ##### [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested with 9.2)

- ##### GPU Compute Capability (tested with versions 5.2, 6.1, and 7.0)

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
