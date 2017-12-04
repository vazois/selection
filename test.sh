#!/bin/bash

START_N=$((1024 * 1024))
END_N=$((1024 * 1024))
START_D=8
END_D=8

MAX_VALUE=100
distr=i

for (( n=$START_N; n<=$END_N; n*=2 ))
do
	for (( d=$START_D; d<=$END_D; d+=2 ))
	do
		fname='d_'$n'_'$d'_'$MAX_VALUE
		#echo "Processing ... "$fname
		if [ ! -f data/$fname ]; then
    		echo "Creating file <"$fname">"
			cd data/; python selectivityGen.py $n $fname $MAX_VALUE $d ; cd ..
		fi
		
		echo "Processing ... <"$fname">"
		./gpu_run -f=data/$fname -mx=$MAX_VALUE
		
	done
	
done