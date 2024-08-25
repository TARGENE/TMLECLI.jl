# Targeted Minimum Loss Based Estimation

This is the main script in this package, it provides a command line interface for the estimation of statistical parameters using targeted Learning.

## Usage

```bash
tmle tmle --help
```

```@docs
tmle
```

## Specifying Estimands

The easiest way to create an estimands' file is to use the companion Julia [TMLE.jl](https://targene.github.io/TMLE.jl/stable/) package and create a `Configuration` structure. This structure can be serialized to a file using any of `serialize` (Julia serialization format), `write_json` (JSON) or `write_yaml` (YAML).

Alternatively you can write this file manually. The following example illustrates the creation of three estimands in YAML format: an Average Treatment Effect (ATE), an Average Interaction Effect (AIE) and a Counterfactual Mean (CM).

```yaml
type: "Configuration"
estimands:
  - outcome_extra_covariates:
      - C1
    type: "AIE"
    treatment_values:
      T1:
        control: 0
        case: 1
      T2:
        control: 0
        case: 1
    outcome: Y1
    treatment_confounders:
      T2:
        - W21
        - W22
      T1:
        - W11
        - W12
  - outcome_extra_covariates: []
    type: "ATE"
    treatment_values:
      T1:
        control: 0
        case: 1
      T3:
        control: "CC"
        case: "AC"
    outcome: Y3
    treatment_confounders:
      T1:
        - W
      T3:
        - W
  - outcome_extra_covariates: []
    type: "CM"
    treatment_values:
      T1: "CC"
      T3: "AC"
    outcome: Y3
    treatment_confounders:
      T1:
        - W
      T3:
        - W
```

## Specifying Estimators

There are two ways the estimators can be specified, either via a plain Julia file or via a configuration string.

### Estimators From A String

An estimator can be described from 3 main axes, depending on:

1. Whether they use cross-validation (sample-splitting) or not.
2. The semi-parametric estimator type: TMLE, wTMLE, OSE.
3. The models used to learn the nuisance functions.

The estimator type and cross-validation scheme are described at once by any of the following

| Estimator's Short Name | Estimator's Description |
| :--------: | :-------: |
| tmle       | Canonical Targeted Minimum-Loss Estimator |
| wtmle      | Canonical Targeted Minimum-Loss Estimator with weighted Fluctuation  |
| ose        | Canonical One-Step Estimator |
| cvtmle     | Cross-Validated Targeted Minimum-Loss Estimator |
| cvwtmle    | Cross-Validated Targeted Minimum-Loss Estimator with weighted Fluctuation  |
| cvose      | Cross-Validated One-Step Estimator |

And the available models are

| Model's Short Name | Model's Description |
| :--------:   | :-------: |
| glm           | A Generalised Linear Model |
| glmnet        | A Cross-Validated Generalised Linear Model |
| xgboost       | The default XGBoost model using the `hist` strategy. |
| tunedxgboost  | A cross-validated grid of XGBoost models across (max_depth, eta) hyperparameters. |
| sl            | A Super Learning strategy using a glmnet, a glm and a grid of xgboost models as in tunedxgboost. |

Then, a configuration string describes the estimators and models in the following way: ESTIMATORS--Q_MODEL--G_MODEL.

- The `ESTIMATORS` substring comprises one or more estimators separated by a single dash, e.g. `cvtmle-ose`. If multiple estimators are specified they will be used sequentially and an estimation result will provide key-value pairs of ESTIMATOR => ESTIMATE.
- The optional `G_MODEL` substring corresponds to the model used to learn the propensity score models. If it is not provided, it will default to the model provided for `Q_MODEL`.
- The optional `Q_MODEL` substring corresponds to the model used to learn the outcome models, it defaults to `glmnet`.

It is probably easier to understand with some examples.

#### Examples

- `tmle--sl--glm`: A single estimator (TMLE) using a Super Learner for the outcome models and a GLM for the propensity score models.
- `cvtmle-ose--xgboost`: Two estimators (CV-TMLE and OSE) using XGBoost for the outcome models and the default strategy for the propensity score models.
- `cvwtmle-cvose`: Two estimators (CV-wTMLE and CV-OSE) using default strategies for both outcome models and propensity score models.
  
#### Note on Cross-Validation

Some of the aforementioned estimators and models use cross-validation under the hood. In this case this using a stratified 3-folds cross-validation where the stratification occurs across both the outcome and treatment variables.

#### Note on GLM and GLMNet

Linear models typically do not involve any interaction terms. Here, to add extra flexibility, both GLM and GLMNet comprise pairwise interaction terms between treatment variables and all other covariates.

### Estimators Via A Julia File

Building an estimator via a configuration string is quite flexible and should cover most use cases. However, in some cases you may want to have full control over the estimation procedure. This is possible by instead providing a Julia configuration file describing the estimators to be used. The file should define an `ESTIMATORS` NamedTuple describing the estimators to be used, and some examples can be found [here](https://github.com/TARGENE/TMLECLI.jl/tree/treatment_values/estimators-configs).

For further information, we recommend you have a look at both:

- [TMLE.jl](https://targene.github.io/TMLE.jl/stable/): The Julia package on which this command line interface is built.
- [MLJ](https://juliaai.github.io/MLJ.jl/dev/): The Julia package used for machine-learning throughout.

## Note on Outputs

We can output results in three different formats: HDF5, JSON and JLS. By default no output is written, so you need to specify at least one. An output can be generated by specifying an output filename for it. For instance `--outputs.json.filename=output.json` will output a JSON file. Note that you can generate multiple formats at once, e.g. `--outputs.json.filename=output.json --outputs.hdf5.filename=output.hdf5` will output both JSON and HDF5 result files. Another important output option is the `pval_threshold`. Each estimation result is accompanied by an influence curve vector and by default these vectors are erased before saving the results because they typically take up too much space and are not usually needed. In some occasions you might want to keep them and this can be achieved by specifiying the output's `pval_threhsold`. For instance `--outputs.hdf5.pval_threshold=1.` will keep all such vectors because all p-values lie in between 0 and 1.

In order to run sieve variance plateau correction after a TMLE run you need to save the results in HDF5 format with influence curve vectors. Furthermore, you will need to save the sample-ids associated with each result. A complete option set for this could be: `--outputs.hdf5.filename=output.hdf5 --outputs.hdf5.pval_threshold=0.05 --sample_ids=true`. In this case, only those results with an individual p-value of less than ``0.05`` will keep track of their influence curves and be considered for sieve variance correction.

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

- The Julia script can be found at [experiments/runtime.jl](https://github.com/TARGENE/TMLECLI.jl/tree/main/experiments/runtime.jl).
- The various estimators used below are further described in the[estimators-configs](https://github.com/TARGENE/TMLE.jl/tree/main/estimators-configs) folder.

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

| Estimator | Unit runtime (s) | Extrapolated runtime to 1000 outcomes |
| --- | :---: | :---: |
| `glm.` | 4.65 | ≈ 1h20 |
| `glmnet` | 7.19 | ≈ 2h |
| `G-superlearning-Q-glmnet` | 50.05| ≈ 13h45 |
| `superlearning` | 168.98 | ≈ 46h |

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
| `glm` | 5.64 | 6.14 | ≈ 6h30 |
| `glmnet` | 17.46 | 22.24 | ≈ 22h |
| `G-superlearning-Q-glmnet` | 430.54 | 438.67 | ≈ 20 days |
| `superlearning` | 511.26 | 567.72 | ≈ 24 days |

We can see that modern high performance computing platforms definitely enable this study design when using GLMs or GLMNets. It is unlikely however, that you will be able to use Super Learning for any of `P(V|W)` or `E[Y|V, W]` if you don't have privileged access to such platform. While the double robustness guarantees will generally not be satisfied, our estimate will still be targeted, which means that its bias will be reduced compared to classic inference using a parametric model.
