
evotree = EvoTreeClassifier(nrounds=10)

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
    interaction_glmnet = Pipeline(
      interaction_transformer = InteractionTransformer(order=3),
      glmnet                  = GLMNetRegressor(),
      cache                   = true
    ),
    evo_1              = EvoTreeRegressor(nrounds=10, lambda=0., gamma=0.3),
    evo_2              = EvoTreeRegressor(nrounds=10, lambda=1., gamma=0.3),
    evo_3              = EvoTreeRegressor(nrounds=20, lambda=0., gamma=0.3),
    evo_4              = EvoTreeRegressor(nrounds=20, lambda=1., gamma=0.3),
    constant           = ConstantRegressor(),
    hal                = HALRegressor(max_degree=1, smoothness_orders=1, num_knots=[10, 5], lambda=10, cv_select=false)
    ),
  # For the estimation of E[Y|W, T]: binary target
  Q_binary = Pipeline(
    interaction_transformer = InteractionTransformer(order=2),
    glmnet                  = GLMNetClassifier(),
    cache                   = false
  ),
  # For the estimation of p(T| W)
  G = TunedModel(
    model = evotree,
    resampling = CV(),
    tuning = Grid(goal=5),
    range = [range(evotree, :max_depth, lower=3, upper=5), range(evotree, :lambda, lower=1e-5, upper=10, scale=:log)],
    measure = log_loss,
    cache=true
    )
)

