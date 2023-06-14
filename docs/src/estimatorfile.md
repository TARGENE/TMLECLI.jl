# Estimator File

TMLE is an adaptive procedure that depends on the specification of learning algorithms for the estimation of the nuisance parameters (see [TMLE.jl](https://targene.github.io/TMLE.jl/stable/) for a description of the assumed setting). In our case, there are two nuisance parameters for which we need to specify learning algorithms:

- `E[Y|T, W, C]`: The mean outcome given the treatment, confounders and extra covariates. It is commonly denoted by `Q` in the Targeted Learning litterature.
- `p(T|W)`: The propensity score. It is commonly denoted by `G` in the Targeted Learning litterature.

## Description of the file

In order to provide maximum flexibility as to the choice of learning algorithms, the estimator file is a plain [Julia](https://julialang.org/) file. It merely must define a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple) called `tmle_spec` with 5 fields as follows:

```julia

tmle_spec = (
  Q_continuous = LinearRegressor(),
  Q_binary     = LogisticClassifier(),
  G            = LogisticClassifier(),
  threshold    = 1e-8,
  cache        = false,
)
```

where:

- `Q_continuous`: is a MLJ model used for the estimation of `E[Y|T, W, C]` when the outcome `Y` is continuous.
- `Q_binary`: is a MLJ model used for the estimation of `E[Y|T, W, C]` when the outcome `Y` is binary.
- `G`: is a MLJ model used for the estimation of `p(T|W)`.
- `threshold`: is the minimum value the propensity score `G` is allowed to take.
- `cache`: controls caching of data by [MLJ machines](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/). Setting it to `true` may result in faster runtime but higher memory usage.

Typically, only `Q_continuous`, `Q_binary` and `threshold` will be adjusted.

## Ready to use estimator files

We recognize not everyone will be familiar with [Julia](https://julialang.org/). We thus provide a set of ready to use estimator files that can be simplified or extended as needed:

- Super Learning: [with](./estimators/superlearning-with-interactions-for-Q.jl) and [without](./estimators/superlearning.jl) interaction terms in the GLM models for Q.
- Super Learning for G and GLMNet for Q: [here](./estimators/G-superlearning-Q-glmnet.jl).
- Super Learning for G and GLM for Q: [here](./estimators/G-superlearning-Q-glm.jl).
- GLMNet: [with](./estimators/glmnet-with-interactions-for-Q.jl) and [without](./estimators/glmnet.jl) interaction terms in the GLM models for Q.
- GLM: [with](./estimators/glm-with-interactions-for-Q.jl) and [without](./estimators/glm.jl) interaction terms in the GLM models for Q.
- XGBoost: [with tuning](./estimators/tuned-xgboost.jl).
