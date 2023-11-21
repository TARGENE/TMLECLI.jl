default_models = TMLE.default_models(
  Q_continuous = LinearRegressor(),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = LogisticClassifier(),
  # This will fail
  G = LogisticClassifier()
)

models = merge(default_models, (T2 = LinearRegressor(),))

ESTIMATORS = (
  TMLE = TMLEE(models=models, weighted=true, ps_lowerbound=0.001),
  OSE  = OSE(models=models)
)