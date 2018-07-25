#!/bin/bash

N=$((128*1024*1024))
D=8

make gpu_cc
./gpu_run -n=$N -d=$D
