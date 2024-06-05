#!/bin/bash

#PBS -N flowsom_benchmarks           ## job name
#PBS -l nodes=1:ppn=16               ## 1 nodes, 4 cores per node
#PBS -l walltime=12:00:00            ## max. 6h of wall time
#PBS -l mem=64gb                    ## 32GB of memory
#PBS -m abe                         ## send mail on abort, begin and end

PIP_DIR="$VSC_SCRATCH/site-packages" # directory to install packages
CACHE_DIR="$VSC_SCRATCH/.cache" # directory to use as cache

# Load PyTorch
module load Python/3.10.4-GCCcore-11.3.0

# activate venv
source venv_joltik/bin/activate







# Start script
cd $PBS_O_WORKDIR/project/tests/benchmarks

rm -rf "$VSC_DATA/output_flowsom"

mkdir "$VSC_DATA/output_flowsom"

#PYTHONPATH="$PYTHONPATH:$PIP_DIR" python exp1.py --quality 20 --only_adv


PYTHONPATH="$PYTHONPATH:$PIP_DIR" python ~/compbio_lloyd/notebooks/test_and_bench.py
