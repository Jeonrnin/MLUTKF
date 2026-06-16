#!/bin/bash
#PBS -N L96EnKF
#PBS -V
#PBS -q mpi
#PBS -W block=false
#PBS -l select=2:ncpus=20
##PBS -l select=2:ncpus=20:mpiprocs=20:ompthreads=1
#PBS -l walltime=06:00:00
#PBS -o job_stdout.log
#PBS -e job_stderr.log

cd $PBS_O_WORKDIR

module purge
module load intel impi py3-mpi4py/4.0.3 py3-intel-numpy/1.15.1 py3-intel-scipy/1.1.0
source /opt/intel/oneapi/setvars.sh --force
export I_MPI_HYDRA_BOOTSTRAP=ssh
export MKL_CBWR=AVX512                      # to ensure the reproducibility for intel-scipy
export OMP_NUM_THREADS=1                    # the number of OpenMP threads for intel-scipy

mpirun -np 40 -ppn 20 python L96enkf.py >& L96enkf.log
#mpirun -np 40 -ppn 20 python L96enkf.py > L96enkf.log 2>&1