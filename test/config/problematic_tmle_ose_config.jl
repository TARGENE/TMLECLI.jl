default_models = TMLE.default_models(
  Q_continuous = LinearRegressor(),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = LogisticClassifier(),
  # This will fail
  G = LogisticClassifier(),
  T2 = LinearRegressor()
)

ESTIMATORS = (
  TMLE = TMLEE(models=default_models, weighted=true, ps_lowerbound=0.001),
  OSE  = OSE(models=default_models)
)