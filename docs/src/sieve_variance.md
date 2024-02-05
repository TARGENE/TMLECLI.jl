# Sieve Variance Plateau Estimation

If the i.i.d. (independent and identically distributed) hypothesis is not satisfied, most of the traditional statistical inference theory falls apart. This is typically possible in population genetics where a study may contain related individuals. Here we leverage a non-parametric method called [Sieve Variance Plateau](https://biostats.bepress.com/ucbbiostat/paper322/) (SVP) estimation. The hypothesis is that the dependence between individuals is sufficiently small, so that our targeted estimator will still be asymptotically unbiased, but its variance will be under estimated. In brief, the SVP estimator computes a variance estimate for a range of thresholds 𝜏, by considering individuals to be independent if their distance exceeds 𝜏. As the distance threshold 𝜏 increases, fewer individuals are assumed to be independent. The maximum of this curve is the most conservative estimate of the variance of the target parameter estimator and constitutes our SVP corrected variance estimator.

## [Usage](@id svp_command)

```bash
tmle sieve-variance-plateau --help
```

```@docs
sieve_variance_plateau
```
