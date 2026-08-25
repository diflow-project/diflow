#! /bin/bash

num=$(cat hostfile | wc -l)

mpirun -n $num --hostfile hostfile nvidia-smi --query-gpu=gpu_name,uuid --format=csv
