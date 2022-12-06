# TargetedEstimation

![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/TARGENE/TargetedEstimation.jl/CI/main?label=Build%20main)
![Codecov branch](https://img.shields.io/codecov/c/github/TARGENE/TargetedEstimation.jl/main?label=Coverage%20main)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/TARGENE/TargetedEstimation.jl)

This package aims at providing a command line interface to run Targeted Maximum Likelihood Estimation.

## Usage

The best way to use the interface is to use the associated [docker image](https://hub.docker.com/r/olivierlabayle/targeted-estimation/tags). Command line arguments can be displayed by:

```bash
julia --project=/TargetedEstimation.jl --startup-file=no scripts/tmle.jl --help
```
