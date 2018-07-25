#!/bin/bash

N=$((32*1024*1024))
D=8

make gpu_cc
./gpu_run -n=$N -d=$D
