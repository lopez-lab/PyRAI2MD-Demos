#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --job-name=hfb-train
#SBATCH --partition=short
#SBATCH --mem=30Gb
#SBATCH --output=%j.o.slurm
#SBATCH --error=%j.e.slurm
#SBATCH --constraint="ib"

export INPUT=input
export WORKDIR=/scratch/lijingbai2009/ChemAccRes/training_curves/train/5_Wigner_and_Wigner_interp
export PYRAIMD=/home/lijingbai2009/share/NN-ChemI/PyRAIMD/bin
export PATH=/work/lopez/Python-3.7.4/bin:$PATH

cd $WORKDIR
python3 $PYRAIMD/PyRAIMD.py $INPUT

