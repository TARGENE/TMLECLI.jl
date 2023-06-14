# TargetedEstimation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://targene.github.io/TargetedEstimation.jl/stable/)
![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/TARGENE/TargetedEstimation.jl/CI.yml?branch=main)
![Codecov](https://img.shields.io/codecov/c/github/TARGENE/TargetedEstimation.jl/main)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/TARGENE/TargetedEstimation.jl)

This package provides two command line interfaces used mainly in the context of TarGene:
1. `scripts/tmle.jl`: To run Targeted Maximum Likelihood Estimation
1. `scripts/sieve_variance.jl`: To run sieve variance correction to account for potential non iid data.

## Usage

The best way to use the command lines is to use the associated [docker image](https://hub.docker.com/r/olivierlabayle/targeted-estimation/tags). Command line arguments can be displayed by:

### tmle.jl

To display command line arguments:

```bash
julia --project=/TargetedEstimation.jl --startup-file=no scripts/tmle.jl --help
```

### sieve_variance.jl

This requires an HDF5 file output by `tmle.jl` and the Genetic Relationship Matrix output by the GCTA software.

To display command line arguments:

```bash
julia --project=/TargetedEstimation.jl --startup-file=no scripts/sieve_variance.jl --help
```

## Experiments

The `experiments` contains various experiments related to genetic association studies: GWAS' and PheWAS'.

### GWAS Runtime

The goal of this experiment is to estimate the running time of TMLE in a GWAS setting. Because the propensity score estimation runtime varies for various SNPs, this is done by running TMLE over 100 SNPs. We estimate the runtime for both a continuous and a binary target and for 4 nuisance parameters specifications: GLM, GLMNet, CrossValidatedXGBoost, Super Learning(GLMNet+CrossValidatedXGBoost). Cross validations selections are performed over 3-folds.

- Associated data: Restricted access. On the University of Edinburgh datastore, `/exports/igmm/datastore/ponting-lab/olivier/misc_datasets/gwas_sample_data.csv`

- Associated script: [experiments/gwas_runtime.jl](experiments/gwas_runtime.jl).

- Julia script usage: `julia --project --startup-file=no experiments/gwas_runtime.jl --help`

- Bash script (to submit jobs on the Eddie cluster): 
    - `qsub experiments/gwas_unit_binary.sh`
    - `qsub experiments/gwas_unit_continuous.sh`

### PheWAS Runtime

The goal of this experiment is to estimate the running time of TMLE in a PheWAS setting. Since the propensity score is estimated only once, it is not driving runtime. The PheWAS is perfomed over more than 760 traits and for 4 nuisance parameters specifications: GLM, GLMNet, CrossValidatedXGBoost, Super Learning(GLMNet+CrossValidatedXGBoost). Cross validations selections are performed over 3-folds.

- Associated data: Restricted access. On the University of Edinburgh datastore, `/exports/igmm/datastore/ponting-lab/olivier/misc_datasets/sample_ukb_data.csv`

- Associated script: [experiments/phewas_runtime.jl](experiments/phewas_runtime.jl).

- Julia script usage: `julia --project --startup-file=no experiments/phewas_runtime.jl --help`

- Bash scripts (to submit jobs on the Eddie cluster):
    - `qsub experiments/phewas_glm.sh`
    - `qsub experiments/phewas_glmnet.sh`
    - `qsub experiments/phewas_xgboost.sh`
    - `qsub experiments/phewas_sl.sh`
