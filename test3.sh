#!/bin/bash

#N=$((1*1024*1024))
#D=8

#make gpu_cc
#./gpu_run -n=$N -d=$D

make gpu_cc
START_N=$((256*1024*1024))
END_N=$((256*1024*1024))
D=8

for (( N=$START_N; N<=$END_N; N*=2 ))
do
	echo "Input: "$N
	./gpu_run -n=$N -d=$D > $N".out"
	#./gpu_run -n=$N -d=$D
done

