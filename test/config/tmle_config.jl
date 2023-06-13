tmle_spec = (
  # Controls caching of data by MLJ machines: turning to `true`` may result in faster execution but higher memory usage
  cache=false,
  # Propensity score threshold
  threshold    = 0.001,
  # For the estimation of E[Y|W, T]: continuous target
  Q_continuous = Stack(
    metalearner        = LinearRegressor(fit_intercept=false),
    resampling         = CV(nfolds=2),
    interaction_glmnet = RestrictedInteractionGLMNetRegressor(order=3),
    evo_10             = EvoTreeRegressor(nrounds=10),
    evo_20             = EvoTreeRegressor(nrounds=20),
    constant           = ConstantRegressor(),
    hal                = HALRegressor(max_degree=1, smoothness_orders=1, num_knots=[10, 5], lambda=10, cv_select=false)
    ),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = Stack(
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = StratifiedCV(nfolds=2),
    cache              = false,
    interaction_glmnet = InteractionGLMNetClassifier(),
    constant           = ConstantClassifier(),
    hal                = HALClassifier(max_degree=1, smoothness_orders=1, num_knots=[10, 5], lambda=10, cv_select=false),
    gridsearch_evo     = GridSearchEvoTreeClassifier(goal=5, nrounds=10, max_depth="3,5", lambda= "1e-5,10,log", resampling=Dict(:type => "CV"))
  ),
  # For the estimation of p(T| W)
  G = Stack(
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = StratifiedCV(nfolds=2),
    interaction_glmnet = RestrictedInteractionGLMNetClassifier(primary_columns = [:T1, :T2], primary_patterns =["C"]),
    constant           = ConstantClassifier(),
    evo                = EvoTreeClassifier(nrounds=10)
  )
)