#!/bin/sh
## script for PyRAIMD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4-23:59:59
#SBATCH --job-name=hfb
#SBATCH --partition=lopez
#SBATCH --mem=11000mb
#SBATCH --output=%j.o.slurm
#SBATCH --error=%j.e.slurm

export INPUT=input
export WORKDIR=/scratch/lijingbai2009/AccChemRes/hfb_permutation/prediction/permut/14
export PYRAIMD=/home/lijingbai2009/share/NN-ChemI/PyRAIMD/bin
export PATH=/work/lopez/Python-3.7.4/bin:$PATH

cd $WORKDIR
python3 $PYRAIMD/PyRAIMD.py $INPUT

