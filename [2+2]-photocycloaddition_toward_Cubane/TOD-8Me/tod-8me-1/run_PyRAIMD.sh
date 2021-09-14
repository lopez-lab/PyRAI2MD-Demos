#!/bin/sh
## script for PyRAIMD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:59:59
#SBATCH --job-name=tod-8mf-1
#SBATCH --partition=large,long,short,lopez
#SBATCH --mem=11000mb
#SBATCH --output=%j.o.slurm
#SBATCH --error=%j.e.slurm

export INPUT=input
export WORKDIR=/scratch/lijingbai2009/R-TOD/github/ML_NAMD_demos/TOD-8Me/tod-8me-1
export PYRAIMD=/home/lijingbai2009/share/NN-ChemI/PyRAIMD/bin
export PATH=/work/lopez/Python-3.7.4/bin:$PATH

cd $WORKDIR
python3 $PYRAIMD/PyRAIMD.py $INPUT

