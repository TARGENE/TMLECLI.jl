# TargetedEstimation

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/TARGENE/TargetedEstimation.jl/CI.yml?branch=main)
![Codecov branch](https://img.shields.io/codecov/c/github/TARGENE/TargetedEstimation.jl/main?label=Coverage%20main)
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

### Speed Test

The goal of this script is to estimate the running time of TMLE in a GWAS setting. This is done by running TMLE over 100 SNPs for both a continuous and a binary target and for 4 nuisance parameters specification:
- Vanilla GLM
- GLMNet cross-validated over 3 folds
- XGBoost cross validated over 3 folds
- Super Learning: GLMNet cross-validated over 3 folds + XGBoost cross validated over 3 folds

The associated julia script is `experiments/speedtest.jl`.

