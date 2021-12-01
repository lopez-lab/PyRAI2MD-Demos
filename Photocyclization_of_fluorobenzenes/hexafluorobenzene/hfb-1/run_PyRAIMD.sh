#!/bin/sh
## script for PyRAIMD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-23:59:59
#SBATCH --job-name=5fb-1
#SBATCH --partition=long,lopez
#SBATCH --mem=11000mb
#SBATCH --output=%j.o.slurm
#SBATCH --error=%j.e.slurm

export INPUT=input
export WORKDIR=/scratch/lijingbai2009/Github_hfb/hexafluorobenzene/hfb-1
export PYRAIMD=/home/lijingbai2009/share/NN-ChemI/PyRAIMD/bin
export PATH=/work/lopez/Python-3.7.4/bin:$PATH

cd $WORKDIR
python3 $PYRAIMD/PyRAIMD.py $INPUT

