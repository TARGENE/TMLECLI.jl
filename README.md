# TargetedEstimation

![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/TARGENE/TargetedEstimation.jl/CI/main?label=Build%20main)
![Codecov branch](https://img.shields.io/codecov/c/github/TARGENE/TargetedEstimation.jl/main?label=Coverage%20main)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/TARGENE/TargetedEstimation.jl)

This package aims at providing a command line interface to run TMLE. The best way to use the interface is to use the associated [docker image](https://hub.docker.com/r/olivierlabayle/targeted-estimation/tags).

## Usage

In the provided docker image, from the project's directory, exact usage can be displayed by:

```bash
julia --project --startup-file=no scripts/tmle.jl --help
```

Here is a description of the arguments:

- `data`: .CSV or .arrow dataset.
- `param-file`: The parameters configuration file
- `estimator-file`: The estimator configuration file
- `out`: Output path for the `.hdf5` or `.csv` file.
- `--save-full`: If this flag is on, `.hdf5` containing the influence curves will be output, otheerwise, only a summary of effect size in `.csv` format is output.
- `--verbosity`: Logging level
