
default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous target
  Q_continuous = LinearRegressor(),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = LogisticClassifier(),
  # For the estimation of p(T| W)
  G = LogisticClassifier()
)

ESTIMATORS = (
  OSE  = OSE(models=default_models),
)