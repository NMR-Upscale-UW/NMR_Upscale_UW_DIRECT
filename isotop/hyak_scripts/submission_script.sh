#!/bin/bash -l
#SBATCH --job-name=generate_data
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=fabdil@uw.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --mem=50G
#SBATCH --export=all
module load foster/python/miniconda/3.8 && \
conda activate /mmfs1/gscratch/stf/fabdil/Capstone/pytorch-cuda11/pytorch-cuda11
export PYTHONPATH=$PYTHONPATH:/mmfs1/gscratch/scrubbed/fabdil/Capstone/pytorch-cuda11/pytorch-cuda11/lib/python3.8/site-packages/
python3 optimize_allcode.py
