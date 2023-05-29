#!/bin/bash
#SBATCH --nodes=2               # Number of nodes
#SBATCH --ntasks=9           # Request 9 CPU cores
#SBATCH --time=00:10:00         # Job should run for up to 5 minutes
#SBATCH --account=msoc  	# Where to charge NREL Hours
#SBATCH --mail-user=Bernard.Knueven@nrel.gov  # If you want email notifications
#SBATCH --mail-type=BEGIN,END,FAIL		 # When you want email notifications
#SBATCH --output=3scen_nofw.%j.out  # %j will be replaced with the job ID

module load conda
module load xpressmp
module load openmpi/4.1.0/gcc-8.4.0

conda activate mpi-sppy 

export OMPI_MCA_btl=self,tcp

cd ${HOME}/software/mpi-sppy/examples/uc

srun python -u -m mpi4py uc_cylinders.py --bundles-per-rank=0 --max-iterations=5 --default-rho=1.0 --num-scens=3 --max-solver-threads=2 --solver-name=xpress_persistent --rel-gap=0.00001 --abs-gap=1 --lagrangian-iter0-mipgap=1e-7 --lagrangian --xhatshuffle --ph-mipgaps-json=phmipgaps.json
