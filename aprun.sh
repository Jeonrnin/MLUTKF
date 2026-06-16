#!/bin/csh -f
#PBS -l select=40:ncpus=1
##PBS -l select=20:ncpus=24
##PBS -l select=10:ncpus=4
##PBS -l select=2:ncpus=24
#PBS -q normal@uri
#PBS -N mpi4py
##PBS -N nobs_lutkf
##PBS -N obs_freq
#PBS -j oe
#PBS -V

cd $PBS_O_WORKDIR

aprun -n 40 -N 1 -S 1 python ./L96enkf.py > test.log
#aprun -n 40 python ./L96enkf.py > test.log
