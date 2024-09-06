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