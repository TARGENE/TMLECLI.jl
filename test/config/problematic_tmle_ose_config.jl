default_models = TMLE.default_models(
  Q_continuous = MLJLinearModels.LinearRegressor(),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = LogisticClassifier(),
  # This will fail
  G = LogisticClassifier(),
  T2 = MLJLinearModels.LinearRegressor()
)

ESTIMATORS = (
  TMLE = Tmle(models=default_models, weighted=true, ps_lowerbound=0.001),
  OSE  = Ose(models=default_models)
)