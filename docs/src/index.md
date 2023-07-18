# TargetedEstimation.jl

The goal of this package, eventually, is to provide a standalone executable to run Targeted Minimum Loss-based Estimation (TMLE) on tabular datasets. It is based on the companion [TMLE.jl](https://targene.github.io/TMLE.jl/stable/) package.

We also provide extensions to the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) universe that are particularly useful in statistical genetics (but not restricted to it):

- Additional [Models](@ref)
- Additional [Resampling Strategies](@ref)

## Installation

The package is not yet registered and must be installed via the url:

```julia
using Pkg; Pkg.add("https://github.com/TARGENE/TargetedEstimation.jl.git") 
```

## Running TMLE

### Run Environment

At this point in time, the package depends on several R dependencies which makes it difficult to package as a single Julia executable. We thus rely on a docker container for that. The main entry point is the `scripts/tmle.jl` script that can be executed in the container provided [here](https://hub.docker.com/r/olivierlabayle/targeted-estimation/tags).

### Example usage

Provided you have the package and all dependencies installed or in the provided docker container, you can run TMLE via the following command:

```bash
julia scripts/tmle.jl DATAFILE PARAMFILE OUTFILE
        --estimator-file=docs/estimators/glmnet.jl
        --hdf5-out=output.hdf5
        --pval-threshold=0.05
        --chunksize=100
        --verbosity=1
```

where:

- `DATAFILE`: A .csv or .arrow file containg the tabular data
- `PARAMFILE`: A serialized [YAML](https://targene.github.io/TMLE.jl/stable/user_guide/#Reading-Parameters-from-YAML-files) or [bin](https://docs.julialang.org/en/v1/stdlib/Serialization/) file containing the estimands to be estimated. The YAML file can be written by hand or programmatically using the [TMLE.parameters_to_yaml](https://targene.github.io/TMLE.jl/stable/api/#TMLE.parameters_to_yaml-Tuple{Any,%20Any}) function.
- `OUTFILE`: The output .csv file
- `--estimator-file`: A .jl file describing the TMLE specifications (see [Estimator File](@ref)).
- `--hdf5-out`: A file to save the influence curves if wanted.
- `--pval-threshold`: Only "significant" (< this threshold) estimates will actually have their influence curves stored in the previous file.
- `--chunksize`: Results are appended to output files in chunks that can be controlled via this option.
- `--verbosity`: THe verbosity level.