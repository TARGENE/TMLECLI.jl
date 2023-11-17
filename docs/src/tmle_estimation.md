# Targeted Minimum Loss Based Estimation

This is the main script in this package, it provides a command line interface for the estimation of statistical parameters using targeted Learning.

## Usage

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

- `DATAFILE`: A CSV (.csv) or Arrow (.arrow) file containing the tabular data. The format will be infered from the extension.
- `PARAMFILE`: A serialized [YAML](https://targene.github.io/TMLE.jl/stable/user_guide/#Reading-Parameters-from-YAML-files) or [bin](https://docs.julialang.org/en/v1/stdlib/Serialization/) file containing the estimands to be estimated. The YAML file can be written by hand or programmatically using the [TMLE.parameters_to_yaml](https://targene.github.io/TMLE.jl/stable/api/#TMLE.parameters_to_yaml-Tuple{Any,%20Any}) function.
- `OUTFILE`: The output .csv file (see [Output file](@ref))
- `--estimator-file`: A Julia file describing the TMLE specifications (see [Estimator File](@ref)).
- `--hdf5-out`: if provided, a path to a file to save the influence curves.
- `--pval-threshold`: Only "significant" (< this threshold) estimates will actually have their influence curves stored in the previous file.
- `--chunksize`: To manage memory, the results are appended to the output files in batches the size of which can be controlled via this option.
- `--verbosity`: The verbosity level.

## Output file

The output file is a plain CSV file containing one line per estimand in the input `PARAMFILE`. The file contains the following columns:

- `PARAMETER_TYPE`: The estimand type (e.g. "ATE", "IATE", ...).
- `TREATMENTS`: A "_&_" separated string containing all treatment variables associated with the estimand.
- `CASE`: A "_&_" separated string containing the treatment variables' case values in the same order as `TREATMENTS`.
- `CONTROL`: A "_&_" separated string containing the treatment variables' control values in the same order as `TREATMENTS`.
- `OUTCOME`: The outcome variable.
- `CONFOUNDERS`: A "_&_" separated string containing the confounding variables.
- `COVARIATES`: A "_&_" separated string containing the extra covariates used to estimate the outcome's mean.
- `INITIAL_ESTIMATE`: The initial estimate before the targeting step.
- `TMLE_ESTIMATE`: The targeted estimate.
- `TMLE_STD`: The standard deviation associated with the targeted estimate.
- `TMLE_PVALUE`: The p-value associated with the targeted estimate.
- `TMLE_LWB`: The 95% confidence interval lower bound associated with the targeted estimate.
- `TMLE_UPB`: The 95% confidence interval upper bound associated with the targeted estimate.
- `ONESTEP_ESTIMATE`: The one step estimate.
- `ONESTEP_STD`: The standard deviation associated with the one step estimate.
- `ONESTEP_PVALUE`: The p-value associated with the one step estimate.
- `ONESTEP_LWB`: The 95% confidence interval lower bound associated with the one step estimate.
- `ONESTEP_UPB`: The 95% confidence interval upper bound associated with the one step estimate.
- `LOG`: A log message if estimation failed.

## Estimator File

TMLE is an adaptive procedure that depends on the specification of learning algorithms for the estimation of the nuisance parameters (see [TMLE.jl](https://targene.github.io/TMLE.jl/stable/) for a description of the assumed setting). In our case, there are two nuisance parameters for which we need to specify learning algorithms:

- `E[Y|T, W, C]`: The mean outcome given the treatment, confounders and extra covariates. It is commonly denoted by `Q` in the Targeted Learning litterature.
- `p(T|W)`: The propensity score. It is commonly denoted by `G` in the Targeted Learning litterature.

### Description of the file

In order to provide maximum flexibility as to the choice of learning algorithms, the estimator file is a plain [Julia](https://julialang.org/) file. This file is optional and omitting it defaults to using generalized linear models. If provided, it must define a [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple) called `tmle_spec` containing any of the following fields as follows (default configuration):

```julia

tmle_spec = (
  Q_continuous = LinearRegressor(),
  Q_binary     = LogisticClassifier(lambda=0.),
  G            = LogisticClassifier(lambda=0.),
  threshold    = 1e-8,
  cache        = false,
  weighted_fluctuation = false
)
```

where:

- `Q_continuous`: is a MLJ model used for the estimation of `E[Y|T, W, C]` when the outcome `Y` is continuous.
- `Q_binary`: is a MLJ model used for the estimation of `E[Y|T, W, C]` when the outcome `Y` is binary.
- `G`: is a MLJ model used for the estimation of `p(T|W)`.
- `threshold`: is the minimum value the propensity score `G` is allowed to take.
- `cache`: controls caching of data by [MLJ machines](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/). Setting it to `true` may result in faster runtime but higher memory usage.
- `weighted_fluctuation`: controls whether the fluctuation for `Q` is a weighted glm or not. If some of the treatment values are rare it may lead to more robust estimation.

Typically, `Q_continuous`, `Q_binary` and `G` will be adjusted and other fields can be left unspecified.

### Ready to use estimator files

We recognize not everyone will be familiar with [Julia](https://julialang.org/). We thus provide a set of ready to use estimator files that can be simplified or extended as needed:

- Super Learning: [with](./estimators/superlearning-with-interactions-for-Q.jl) and [without](./estimators/superlearning.jl) interaction terms in the GLM models for Q.
- Super Learning for G and GLMNet for Q: [here](./estimators/G-superlearning-Q-glmnet.jl).
- Super Learning for G and GLM for Q: [here](./estimators/G-superlearning-Q-glm.jl).
- GLMNet: [with](./estimators/glmnet-with-interactions-for-Q.jl) and [without](./estimators/glmnet.jl) interaction terms in the GLM models for Q.
- GLM: [with](./estimators/glm-with-interactions-for-Q.jl) and [without](./estimators/glm.jl) interaction terms in the GLM models for Q.
- XGBoost: [with tuning](./estimators/tuned-xgboost.jl).

## Runtime

Targeted Learning can quickly become computationally intensive compared to traditional parametric inference. Here, we illustrate typical runtimes using examples from population genetics. This is because population genetics is currently the main use case for this package, but it shouldn't be understood as the only scope. In fact, the two most prominent study designs in population genetics are perfect illustrations of the computational complexity associated with Targeted Learning.

### Preliminary

Remember that for each estimand of interest, Targeted Learning requires 3 main ingredients that drive computational complexity:

- An estimator for the propensity score: `G(T, W) = P(T|W)`.
- An estimator for the outcome's mean: `Q(T, W) = E[Y|T, W]`.
- A targeting step towards the estimand of interest.

While the targeting step has a fixed form, both `G` and `Q` require specification of learning algorithms that can range from simple generalized linear models to complex Super Learners. In general, one doesn't know how the data has been generated and the model space should be kept as large as possible in order to provide valid inference. This means we recommend the use Super Learning for both `G` and `Q` as it comes with asymptotic theoretical guarantees. However, Super Learning is an expensive procedure and, depending on the context, might become infeasible. Also, notice that while the targeting step is specific to a given estimand, `G` and `Q` are only specific to the variables occuring in the causal graph. This means that they can potentially be cleverly reused across the estimation of multiple estimands. Note that this clever reuse, is already baked into this package, and nothing needs to be done beside specifying the learning algorithms for `G` and `Q`. The goal of the subsequent sections is to provide some examples, guiding the choice of those learning algorithms.

In what follows, `Y` is an outcome of interest, `W` a set of confounding variables and `T` a genetic variation. Genetic variations are usually represented as a pair of alleles corresponding to an individual's genotype. We will further restrict the scope to bi-allelic single nucleotide variations. This means that, at a given locus where the two alleles are `A` and `C`, an individual could have any of the following genotype: `AA`, `AC`, `CC`. Those will be our treatment values.

For all the following experiments:

- The Julia script can be found at [experiments/runtime.jl](../../experiments/runtime.jl).
- The various estimators used below are further described in [Ready to use estimator files](@ref).

### Multiple treatment contrasts

In a classic randomized control trial, the treatment variable can only take one of two levels: `treated` or `not treated`. In out example however, any genetic variation takes its values from three different levels. As such, the `treated` and `not treated` levels need to be defined and any of the following contrasts can be of interest:

- `AA` -> `AC`
- `AC` -> `CC`
- `AA` -> `CC`

For a given outcome and genetic variation, for each contrast, both `G` and `Q` are actually invariant. This shows a first level of reduction in computational complexity. **Both `G` and `Q` need to be fitted only once across multiple treatment contrasts and only the targeting step needs to be carried out again.**

### The PheWAS study design

In a PheWAS, one is interested in the effect of a genetic variation across many outcomes (typically around 1000). Because the treatment variable is always the same, the propensity score `G` can be reused across all parameters, which drastically reduces computational complexity.

```@raw html
<div style="text-align:center">
<img src="assets/phewas.png" alt="PheWAS" style="width:400px;"/>
</div>
```

With this setup in mind, the computational complexity is mostly driven by the specification of the learning algorithms for `Q`, which will have to be fitted for each outcome. For 10 outcomes, we estimate the 3 Average Treatment Effects corresponding to the 3 possible treatment contrasts defined in the previous section. There are thus two levels of reuse of `G` and `Q` in this study design. In the table below are presented some runtimes for various specifications of `G` and `Q` using a single cpu. The "Unit runtime" is the average runtime across all estimands and can roughly be extrapolated to bigger studies.

| Estimator file | Unit runtime (s) | Extrapolated runtime to 1000 outcomes |
| --- | :---: | :---: |
| `docs/src/estimators/glm.jl` | 4.65 | ≈ 1h20 |
| `docs/src/estimators/glmnet.jl` | 7.19 | ≈ 2h |
| `docs/src/estimators/G-superlearning-Q-glmnet.jl` | 50.05| ≈ 13h45 |
| `docs/src/estimators/superlearning.jl` | 168.98 | ≈ 46h |

Depending on the exact setup, this means one can probably afford to use Super Learning for at least the estimation of `G` (and potentially also for `Q` for a single PheWAS). This turns out to be a great news because TMLE is a double robust estimator. As a reminder, it means that only one of the estimators for `G` or `Q` needs to converge sufficiently fast to the ground truth to guarantee that our estimates will be asymptotically unbiased.

Finally, note that those runtime estimates should be interpreted as worse cases, this is because:

- Only 1 cpu is used.
- Most modern high performance computing platform will allow further parallelization.
- In the case where `G` only is a Super Learner, since the number of parameters is still relatively low in this example, it is possible that the time to fit `G` still dominates the runtime.
- Runtimes include precompilation which becomes negligible with the size of the study.

### The GWAS study design

In a GWAS, the outcome variable is held fixed and we are interested in the effects of very many genetic variations on this outcome (typically 800 000 for a genotyping array). The propensity score cannot be reused across parameters resulting in a more expensive run.

```@raw html
<div style="text-align:center">
<img src="assets/gwas.png" alt="GWAS" style="width:400px;"/>
</div>
```

Again, we estimate the 3 Average Treatment Effects corresponding to the 3 possible treatment contrasts. However we now look at 3 different genetic variations and only one outcome. In the table below are presented some runtimes for various specifications of `G` and `Q` using a single cpu. The "Unit runtime" is the average runtime across all estimands and can roughly be extrapolated to bigger studies.

| Estimator file | Continuous outcome unit runtime (s) | Binary outcome unit runtime (s) | Projected Time on HPC (200 folds //) |
| --- | :---: | :---: | :---: |
| `docs/src/estimators/glm.jl` | 5.64 | 6.14 | ≈ 6h30 |
| `docs/src/estimators/glmnet.jl` | 17.46 | 22.24 | ≈ 22h |
| `docs/src/estimators/G-superlearning-Q-glmnet.jl` | 430.54 | 438.67 | ≈ 20 days |
| `docs/src/estimators/superlearning.jl` | 511.26 | 567.72 | ≈ 24 days |

We can see that modern high performance computing platforms definitely enable this study design when using GLMs or GLMNets. It is unlikely however, that you will be able to use Super Learning for any of `P(V|W)` or `E[Y|V, W]` if you don't have privileged access to such platform. While the double robustness guarantees will generally not be satisfied, our estimate will still be targeted, which means that its bias will be reduced compared to classic inference using a parametric model.
