# TargetedEstimation.jl

The goal of this package, eventually, is to provide a standalone executable to run Targeted Minimum Loss-based Estimation ([TMLE](https://link.springer.com/book/10.1007/978-1-4419-9782-1)) on large scale tabular datasets. It is based on the companion [TMLE.jl](https://targene.github.io/TMLE.jl/stable/) package.

The various command line interfaces are described in the following sections:

- [Targeted Minimum Loss Based Estimation](@ref): The main command line interface provided in this project to run TMLE.
- [Sieve Variance Plateau Estimation](@ref): Variance correction for non i.i.d. data.

We also provide extensions to the [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) universe that are particularly useful in statistical genetics (but not restricted to it):

- Additional [Models](@ref)
- Additional [Resampling Strategies](@ref)
