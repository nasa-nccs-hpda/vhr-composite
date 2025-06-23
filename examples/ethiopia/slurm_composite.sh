#!/bin/bash
#SBATCH --job-name "composite"
#SBATCH --time=05-00:00:00
#SBATCH -G 1
#SBATCH -c10
#SBATCH --mem-per-cpu=10240
#SBATCH --mail-user=jordan.a.caraballo-vega@nasa.gov
#SBATCH --mail-type=ALL

module load singularity

echo Starting $1

date

singularity exec --env \
    PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite-jordan-edits:/explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow" \
    --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects,/panfs/ccds02/nobackup/projects \
    /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python landcover_composite_pipeline.py \
    -c composite_ethiopia_epoch1.yaml -t test_tile_0_test.txt -s composite

date

echo Done $1
