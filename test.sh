#!/bin/bash

N=$((128*1024 * 1024))
D=8
MAX_VALUE=100
#AND=0#OR=1
type=1

fname='d_'$N'_'$D'_'$MAX_VALUE
if [ ! -f data/$fname ]; then
 	echo "Creating file <"$fname">"
	cd data/; python selectivityGen.py $N $fname $MAX_VALUE $D ; cd ..
fi

args=args.out
cd data/; python genQuerynums.py $MAX_VALUE 8 $type > ../args.out ; cd ..

make gpu_cc
while IFS='' read -r line || [[ -n "$line" ]]; do
    #echo "Text read from file: $line"
    #echo "$line"
    ./gpu_run -f=data/$fname -t=$type $line
    #exit 1
done < "args.out"