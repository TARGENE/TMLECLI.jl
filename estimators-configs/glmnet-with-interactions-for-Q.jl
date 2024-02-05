default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous outcome
  Q_continuous = Pipeline(
    RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
    GLMNetRegressor(resampling=CV(nfolds=3)),
    cache = false
  ),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary    = Pipeline(
    RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
    GLMNetClassifier(resampling=StratifiedCV(nfolds=3)),
    cache = false
  ),
  # For the estimation of p(T| W)
  G           = GLMNetClassifier(resampling=StratifiedCV(nfolds=3))
)

ESTIMATORS = (
  TMLE = TMLEE(models=default_models, weighted=true, ps_lowerbound=1e-8),
)