default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous outcome
  Q_continuous = MLJLinearModels.LinearRegressor(),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = LogisticClassifier(lambda=0.),
  # For the estimation of p(T| W)
  G = LogisticClassifier(lambda=0.)
)

ESTIMATORS = (
  TMLE = Tmle(models=default_models, weighted=true, ps_lowerbound=1e-8),
)