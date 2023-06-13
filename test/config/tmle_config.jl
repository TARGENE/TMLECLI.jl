evotree = EvoTreeClassifier(nrounds=10)

tmle_spec = (
  # Controls caching of data by MLJ machines: turning to `true`` may result in faster execution but higher memory usage
  cache=false,
  # Propensity score threshold
  threshold    = 0.001,
  # For the estimation of E[Y|W, T]: continuous target
  Q_continuous = Stack(
    metalearner        = LinearRegressor(fit_intercept=false),
    resampling         = CV(nfolds=2),
    interaction_glmnet = Pipeline(
      interaction_transformer = RestrictedInteractionTransformer(order=3, primary_variables_patterns=[r"^rs[0-9]+"]),
      glmnet                  = GLMNetRegressor(),
      cache                   = false
    ),
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
    interaction_glmnet = Pipeline(
      interaction_transformer = InteractionTransformer(order=2),
      glmnet                  = GLMNetClassifier(),
      cache                   = false
    ),
    constant           = ConstantClassifier(),
    hal                = HALClassifier(max_degree=1, smoothness_orders=1, num_knots=[10, 5], lambda=10, cv_select=false),
    gridsearch_evo     = TunedModel(
      model = evotree,
      resampling = CV(),
      tuning = Grid(goal=5),
      range = [range(evotree, :max_depth, lower=3, upper=5), range(evotree, :lambda, lower=1e-5, upper=10, scale=:log)],
      measure = log_loss,
      cache=false
      )
  ),
  # For the estimation of p(T| W)
  G = Stack(
    metalearner        = LogisticClassifier(lambda=0., fit_intercept=false),
    resampling         = StratifiedCV(nfolds=2),
    interaction_glmnet = Pipeline(
      interaction_transformer = RestrictedInteractionTransformer(
          order=2,
          primary_variables=[:T1, :T2],
          primary_variables_patterns=[r"C"]
          ),
      glmnet=GLMNetRegressor(),
      cache=false
    ),
    constant           = ConstantClassifier(),
    evo                = EvoTreeClassifier(nrounds=10)
  )
)