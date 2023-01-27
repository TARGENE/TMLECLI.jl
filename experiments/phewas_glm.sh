#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N PheWASGLM
#$ -wd /home/s2042526/UK-BioBank-53116/users/olivier/dev/TargetedEstimation.jl
#$ -l h_vmem=4G
#$ -pe sharedmem 8
source ~/.bashrc
julia --project --threads=8 --startup-file=no experiments/phewas_runtime.jl /home/s2042526/UK-BioBank-53116/users/olivier/data/sample_ukb_data.csv --strategy=glm
