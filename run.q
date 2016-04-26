#!/bin/bash

#PBS -l nodes=1:ppn=2
#PBS -l walltime=20:00:00
#PBS -l mem=20GB
#PBS -N a4
#PBS -j oe
module purge

module load torch-deps/7
module load torch/intel/20151009
cd /scratch/cdg356/deeplearning/DLA4
th loop_lstm_gru.lua
th loop_dropout.lua
