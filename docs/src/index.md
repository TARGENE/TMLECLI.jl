# TargetedEstimation.jl

The goal of this package, eventually, is to provide a standalone executable to run large scale Targeted Minimum Loss-based Estimation ([TMLE](https://link.springer.com/book/10.1007/978-1-4419-9782-1)) on tabular datasets. To learn more about TMLE, please visit [TMLE.jl](https://targene.github.io/TMLE.jl/stable/), the companion package.

The various command line interfaces provided here are described in the following sections and can be run in the associated [Docker container](https://hub.docker.com/r/olivierlabayle/targeted-estimation/tags):

- [Targeted Minimum Loss Based Estimation](@ref): The main command line interface provided in this project to run TMLE.
- [Sieve Variance Plateau Estimation](@ref): Variance correction for non i.i.d. data.

We also provide extensions to the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) universe that are particularly useful in statistical genetics (but not restricted to it):

- Additional [Models](@ref)
- Additional [Resampling Strategies](@ref)
