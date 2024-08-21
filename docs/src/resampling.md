# Resampling Strategies

```@meta
CurrentModule = TMLECLI
```

We also provide additional resampling strategies compliant with the `MLJ.ResamplingStrategy` interface.

## AdaptiveResampling

The AdaptiveResampling strategies will determine the number of cross-validation folds adaptively based on the available data. This is inspired from the [this paper](https://academic.oup.com/ije/advance-article/doi/10.1093/ije/dyad023/7076266) on practical considerations for super learning.

The `AdaptiveCV` will determine the number of folds adaptively and perform a classic cross-validation split:

```@docs
AdaptiveCV
```

The `AdaptiveStratifiedCV` will determine the number of folds adaptively and perform a stratified cross-validation split:

```@docs
AdaptiveStratifiedCV
```

## JointStratifiedCV

Sometimes, the treatment variables (or some other features) are imbalanced and naively performing cross-validation or stratified cross-validation could result in the violation of the positivity hypothesis. To overcome this difficulty, the following `JointStratifiedCV`, performs a stratified cross-validation based on both features variables and the outcome variable.

```@docs
JointStratifiedCV
```
