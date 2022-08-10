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

- `treatments`, `targets`, `confounders`: Those are CSV files containing the data. An extra "SAMPLE_ID" column is required to identify each row. Note that all treatments variables should be interpretable as categorical variables.
- `parameters-file`: The parameters configuration file described in the [Parameters configuration file section](#parameters-configuration-file).
- `estimator-file`: The estimator configuration file described in the [Estimator configuration file section](#estimator-configuration-file).
- `out`: Output path for the `.hdf5` file.
- `--covariates`: Currently unused because the TMLE package cannot yet make use of it (see [this issue](https://github.com/olivierlabayle/TMLE.jl/issues/59)).
- `--target-type`: The script can only process one target type at a time which can be either `Bool` or `Real`.
- `--save-full`: Whether nuisance parameters should also be saved.

## Parameters configuration file

A Parameters configuration file, describes the causal model and the parameters of interest. It contains up to 5 sections: 4 sections to describe the variables' roles in the causal model and 1 section for the parameters. Examples of such configuration files can be found in either [test/config/ate_parameters.yaml](test/config/ate_parameters.yaml), [test/config/iate_parameters.yaml](test/config/iate_parameters.yaml) or [test/config/iate_parameters_only_cont_1.yaml](test/config/iate_parameters_only_cont_1.yaml).

The variables sections are `Targets`, `Treatments`, `Confounders` and `Covariates`. `Treatments`, `Confounders` and `Covariates` will be fixed during the run while `Targets` will be looped through. Those sections are all optional and, if not provided, all the variables in the associated files will be loaded. If provided, they should be a list of variables.

The `Parameters` section lists all parameters of interest. Each parameter has a `name` and subsections corresponding to each treatment variable. Each of those subsections should provide a case and control value for the treatment.

## Estimator configuration file

The estimator configuration file describes the TMLE specification for the estimation of the parameters defined in the [Parameters configuration file](#parameters-configuration-file). Examples of such configuration files can be found in either [test/config/tmle_config.yaml](test/config/tmle_config.yaml) or [test/config/tmle_config_2.yaml](test/config/tmle_config_2.yaml)

This configuration contains 4 sections:

- The `threshold` section is a simple floating point number that specifies the minimum allowed value for p(Treatments|Confounders).
- The `Q_binary`, `Q_continuous` and `G` sections describe the learners for the nuisance parameters. Each of them contains a `model` that corresponds to a valid MLJ model constructor and further keyword hyperparameters. For instance, a Stack can be provided a `measures` argument to evaluate internal algorithms during cross validation. It can also be provided a potentially adaptive `resampling` strategy and the library of `models`. Each of those models can specify a grid of hyperparameters that will individually define a learning algorithm.
  - `Q_binary` corresponds to E[Target| Confounders, Covariates] when the targets are binary variables.
  - `Q_continuous` corresponds to E[Target| Confounders, Covariates] when the targets are continuous variables.
  - `G` corresponds to the joint distribution p(Treatments|Confounders).
