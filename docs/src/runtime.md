# Runtime

Targeted Learning can quickly become computationally intensive compared to traditional parametric inference. Here, we illustrate typical runtimes using examples from population genetics. This is because population genetics is currently the main use case for this package, but it shouldn't be understood as the only scope. In fact, the two most prominent study designs in population genetics are perfect illustrations of the computational complexity associated with Targeted Learning.

## Preliminary

Remember that for each estimand of interest, Targeted Learning requires 3 main ingredients that drive computational complexity:

- The estimation of the propensity score: `G(T, W) = P(T|W)`.
- The estimation of the outcome's mean: `Q(T, W) = E[Y|T, W]`.
- The targeting step towards the estimand of interest.

While the targeting step has a fixed form, Both `G` and `Q` require specification of learning algorithms that can range from simple generalized linear models to complex Super Learners. In general, one doesn't know how the data has been generated and the model space should be kept as large as possible in order to provide valid inference. This means we recommend to use Super Learning for both `G` and `Q` as it comes with asymptotic theoretical guarantees. However, Super Learning is an expensive procedure and, depending on the context, might become infeasible. Also, notice that while the targeting step is specific to a given estimand, `G` and `Q` are only specific to the variables occuring in the causal graph. This means that they can potentially be cleverly reused across the estimation of multiple estimands. Note that this clever reuse, is already baked into this package, and nothing needs to be done beside specifying the learning algorithms for `G` and `Q`. The goal of the subsequent sections is to provide some examples, guiding the choice of those learning algorithms.

## The PheWAS study design

In a PheWAS, the treatment variable is a genetic variant, it is held fixed across the study. We are interested in the effect of this variant on various outcome variables (typically around 1000). Because the treatment variable is always the same, the propensity score `G` can be reused across all parameters, which drastically reduces computational complexity.

```@raw html
<div style="text-align:center">
<img src="assets/phewas.png" alt="PheWAS" style="width:400px;"/>
</div>
```

With this setup in mind, the complexity is mostly driven by the specification of the learning algorithms `Q`, which will have to be fitted for each outcome. In the table below are presented some runtimes for various specifications of `Q` on 100 outcomes.

| Script | Estimator file | Runtime across 100 outcomes |
| --- | --- | :---: |
| --- | --- | :---: |

Depending on the exact setup, this means one can probably afford to use Super Learning for at least the estimation of `G` (and potentially also for `Q` for a single PheWAS). This turns out to be a great news because TMLE is a double robust estimator. As a reminder, it means that only one of the estimators for `G` or `Q` needs to converge sufficiently fast to the ground truth to guarantee that our estimates will be asymptotically unbiased.

## The GWAS study design

In a GWAS, the outcome variable is held fixed and we are interested in the effects of very many genetic variants on this outcome (typically 800 000 for a genotyping array). The propensity score cannot be reused across parameters resulting in a more expensive run.

```@raw html
<div style="text-align:center">
<img src="assets/gwas.png" alt="GWAS" style="width:400px;"/>
</div>
```

It is thus quite unlikely that you will be able to use Super Learning for any of `P(V|W)` or `E[Y|V, W]`. And thus, no double robustness guarantee will be satisfied in general. However, our estimate will still be targeted, which means that its bias will be reduced compared to classic inference using a parametric model.

## Note on multiple treatment contrasts

In a classic randomized control trial, the treatment variable can only take one of two levels: `treated` or `not treated`. In observational studies, this is not necessarily the case. For instance, in population genetics, a genetic variant (or genotype), usually takes its values from three different levels. For instance, one could be any of `AA`, `AC` or `CC` at a given locus. As such, the `treated` and `not treated` levels need to be defined and any of the following contrasts can be of interest:

- `AA` -> `AC`
- `AC` -> `CC`
- `AA` -> `CC`

In this situation, both `G` and `Q` can be reused for each contrast and only the targeting step needs to be carried out, thereby saving computations.
