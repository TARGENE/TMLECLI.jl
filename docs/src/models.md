# Models

```@meta
CurrentModule = TMLECLI
```

Because [TMLE.jl](https://targene.github.io/TMLE.jl/stable/) is based on top of [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/), we can support any model respecting the MLJ interface. At the moment, we readily support all models from the following packages:

- [MLJLinearModels](https://juliaai.github.io/MLJLinearModels.jl/stable/): Generalized Linear Models in Julia.
- [XGBoost.jl](https://dmlc.github.io/XGBoost.jl/dev/): Julia wrapper of the famous [XGBoost](https://xgboost.readthedocs.io/en/stable/) package.
- [EvoTrees.jl](https://evovest.github.io/EvoTrees.jl/stable/): A pure Julia implementation of histogram based gradient boosting trees (subset of XGBoost)
- [GLMNet](https://github.com/JuliaStats/GLMNet.jl): A Julia wrapper of the [glmnet](https://glmnet.stanford.edu/articles/glmnet.html) package. See the [GLMNet](@ref) section.
- [MLJModels](https://github.com/JuliaAI/MLJModels.jl): General utilities such as the `OneHotEncoder` or `InteractionTransformer`.

Further support for more packages can be added on request, please fill an [issue](https://github.com/TARGENE/TMLE-CLI.jl/issues).

Also, because the estimator file used by the TMLE CLI is a pure Julia file, it is possible to use it in order to install additional package that can be used to define additional models.

Finally, we also provide some additional models described in [Additional models provided by TMLE-CLI.jl](@ref).

## Additional models provided by TMLE-CLI.jl

### GLMNet

This is a simple wrapper around the `glmnetcv` function from the [GLMNet.jl](https://github.com/JuliaStats/GLMNet.jl) package. The only difference is that the resampling is made based on [MLJ resampling strategies](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/#Built-in-resampling-strategies).

```@docs
GLMNetRegressor(;resampling=CV(), params...)
```

```@docs
GLMNetClassifier(;resampling=StratifiedCV(), params...)
```

### RestrictedInteractionTransformer

This transformer generates interaction terms based on a set of primary variables in order to limit the combinatorial explosion.

```@docs
RestrictedInteractionTransformer
```

### BiAllelicSNPEncoder

This transformer, mostly useful for genetic studies, converts bi-allelic single nucleotide polyphormism columns, encoded as Strings to a count of one of the two alleles.

```@docs
BiAllelicSNPEncoder
```
