#!/bin/bash
declare -a ndim=(21 34 55)
for i in {0..2}
do
	./demo -r 15 -p 200 -d "${ndim[$i]}" -o 1001 -e 15000000 > results/F_"${ndim[$i]}".report
done
