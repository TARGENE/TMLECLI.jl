tmle_spec = (
  # Controls caching of data by MLJ machines: turning to `true`` may result in faster execution but higher memory usage
  cache = true,
  # Propensity score threshold
  threshold    = 1e-8,
  # For the estimation of E[Y|W, T]: continuous target
  Q_continuous = Stack(
    metalearner        = LinearRegressor(fit_intercept=false),
    cache              = true,
    resampling         = AdaptiveCV(CV(nfolds=2)),
    interaction_glmnet = InteractionGLMNetRegressor(order=3, cache=true),
    evo_1              = EvoTreeRegressor(nrounds=10, lambda=0., gamma=0.3),
    evo_2              = EvoTreeRegressor(nrounds=10, lambda=1., gamma=0.3),
    evo_3              = EvoTreeRegressor(nrounds=20, lambda=0., gamma=0.3),
    evo_4              = EvoTreeRegressor(nrounds=20, lambda=1., gamma=0.3),
    constant           = ConstantRegressor(),
    hal                = HALRegressor(max_degree=1, smoothness_orders=1, num_knots=[10, 5], lambda=10, cv_select=false)
    ),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = InteractionGLMNetClassifier(),
  # For the estimation of p(T| W)
  G = GridSearchEvoTreeClassifier(goal=5, nrounds=10, max_depth="3,5", lambda= "1e-5,10,log", resampling=Dict(:type => "CV"), cache=true)
)

