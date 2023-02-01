#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N GWASUnitBinary
#$ -wd /home/s2042526/UK-BioBank-53116/users/olivier/dev/TargetedEstimation.jl
#$ -l h_vmem=3G
#$ -pe sharedmem 8
source ~/.bashrc
julia --project --threads=8 --startup-file=no experiments/gwas_runtime.jl /home/s2042526/UK-BioBank-53116/users/olivier/data/gwas_sample_data.csv --target=binary
