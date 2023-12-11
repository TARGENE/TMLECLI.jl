default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous outcome
  Q_continuous = GLMNetRegressor(resampling=CV(nfolds=3)),
  # For the estimation of E[Y|W, T]: binary outcome
  Q_binary = GLMNetClassifier(resampling=StratifiedCV(nfolds=3)),
  # For the estimation of p(T| W)
  G = GLMNetClassifier(resampling=StratifiedCV(nfolds=3))
)

ESTIMATORS = (
  TMLE = TMLEE(models=default_models, weighted=true, ps_lowerbound=1e-8),
)