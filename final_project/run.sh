#!/bin/bash

#TMPDIR=~ srun --gres=gpu:1 ncu -o ncu_report_2--set full ./main 

#srun --exclusive --gres=gpu:1 nsys profile ./main $@ -v -n 128

srun --exclusive --gres=gpu:1 \
    ./main $@ -v -n 512