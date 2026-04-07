make_regex_from_treatments(treatment_variables) = [Regex(string("^", t, "\$")) for t in treatment_variables]

# Resampling

RESAMPLING(treatment_variables) = JointStratifiedCV(patterns=make_regex_from_treatments(treatment_variables), resampling=StratifiedCV(nfolds=3)) 

# Interaction Transformer

INTERACTION_TRANSFORMER(treatment_variables) = RestrictedInteractionTransformer(order=2, primary_variables_patterns=make_regex_from_treatments(treatment_variables))

# GLM

function GLM_REGRESSOR(treatment_variables, interactions) 
    return if interactions
        Pipeline(
            INTERACTION_TRANSFORMER(treatment_variables), 
            MLJLinearModels.LinearRegressor()
        )
    else
        MLJLinearModels.LinearRegressor()
    end
end

function GLM_CLASSIFIER(treatment_variables, interactions) 
    return if interactions
        Pipeline(
            INTERACTION_TRANSFORMER(treatment_variables), 
            LogisticClassifier(lambda=0.)
        )
    else
        LogisticClassifier(lambda=0.)
    end
end

# GLMNET

function GLMNET_REGRESSOR(treatment_variables, interactions)
    return if interactions
        Pipeline(
            INTERACTION_TRANSFORMER(treatment_variables),
            GLMNetRegressor(resampling=RESAMPLING(treatment_variables)),
            cache = false
        )
    else
        GLMNetRegressor(resampling=RESAMPLING(treatment_variables))
    end
end

function GLMNET_CLASSIFIER(treatment_variables, interactions)
    return if interactions
        Pipeline(
            INTERACTION_TRANSFORMER(treatment_variables),
            GLMNetClassifier(resampling=RESAMPLING(treatment_variables)),
            cache = false
        )
    else
        GLMNetClassifier(resampling=RESAMPLING(treatment_variables))
    end
end

# XGBoost

const XGBOOST_CLASSIFIER = XGBoostClassifier(tree_method="hist")

const XGBOOST_REGRESSOR = XGBoostRegressor(tree_method="hist")

TUNEDXGBOOST_REGRESSOR(treatment_variables, interactions) = TunedModel(
    model = XGBOOST_REGRESSOR,
    resampling = RESAMPLING(treatment_variables),
    tuning = Grid(goal=20),
    range = [
        range(XGBOOST_REGRESSOR, :max_depth, lower=3, upper=7), 
        range(XGBOOST_REGRESSOR, :lambda, lower=1e-5, upper=10, scale=:log)
        ],
    measure = rmse,
    cache=false
)

TUNEDXGBOOST_CLASSIFIER(treatment_variables, interactions) = TunedModel(
    model = XGBOOST_CLASSIFIER,
    resampling = RESAMPLING(treatment_variables),
    tuning = Grid(goal=20),
    range = [
        range(XGBOOST_CLASSIFIER, :max_depth, lower=3, upper=7), 
        range(XGBOOST_CLASSIFIER, :lambda, lower=1e-5, upper=10, scale=:log)
        ],
    measure = log_loss,
    cache=false
)

# Super Learning

const XGBOOST_CLASSIFIER_GRID = (;(Symbol("xgboost_classifier_", id) => XGBoostClassifier(tree_method="hist", max_depth=max_depth, eta=η, num_round=100) 
        for (id, (max_depth, η)) ∈ enumerate(Iterators.product([2, 4, 6, 8], [0.001, 0.01, 0.3])))...)

const XGBOOST_REGRESSOR_GRID = (;(Symbol("xgboost_regressor_", id) => XGBoostRegressor(tree_method="hist", max_depth=max_depth, eta=η, num_round=100) 
    for (id, (max_depth, η)) ∈ enumerate(Iterators.product([2, 4, 6, 8], [0.001, 0.01, 0.3])))...)

SL_REGRESSOR(treatment_variables, interactions) = Stack(;
    metalearner        = MLJLinearModels.LinearRegressor(fit_intercept=false),
    resampling         = RESAMPLING(treatment_variables),
    cache              = false,
    glmnet             = GLMNET_REGRESSOR(treatment_variables, interactions),
    lr                 = GLM_REGRESSOR(treatment_variables, interactions),
    XGBOOST_REGRESSOR_GRID...
)

SL_CLASSIFIER(treatment_variables, interactions) = Stack(;
    metalearner        = LogisticClassifier(lambda=0.,fit_intercept=false),
    resampling         = RESAMPLING(treatment_variables),
    cache              = false,
    glmnet             = GLMNET_CLASSIFIER(treatment_variables, interactions),
    lr                 = GLM_CLASSIFIER(treatment_variables, interactions),
    XGBOOST_CLASSIFIER_GRID...
)

# Parser

function estimator_from_string(estimator_string, models, resampling; prevalence=prevalence, prevalence_file=nothing)
    return if estimator_string == "TMLE"
        Tmle(models=models, weighted=false, prevalence=prevalence, prevalence_file=prevalence_file)
    elseif estimator_string == "WTMLE"
        Tmle(models=models, weighted=true, prevalence=prevalence, prevalence_file=prevalence_file)
    elseif estimator_string == "OSE"
        Ose(models=models)
    elseif estimator_string == "CVTMLE"
        Tmle(models=models, weighted=false, resampling=resampling, prevalence=prevalence, prevalence_file=prevalence_file)
    elseif estimator_string == "CVWTMLE"
        Tmle(models=models, weighted=true, resampling=resampling, prevalence=prevalence,  prevalence_file=prevalence_file)
    elseif estimator_string == "CVOSE"
        Ose(models=models, resampling=resampling)
    else
        throw(ArgumentError(string("Unknown estimator: ", estimator_string)))
    end
end

model_from_string(model_string, treatment_variables; interactions=true) = eval(Symbol(model_string))(treatment_variables, interactions)

function estimators_from_string(;config_string="wtmle-ose", treatment_variables=Set(Symbol[]), prevalence=prevalence, prevalence_file=nothing)
    config_string = uppercase(config_string)
    # Create models
    components = split(config_string, "--")
    if length(components) == 3
        ## There is an estimator specification, a Q specification and a G specification
        q_string = components[2]
        g_string = components[3]
    elseif length(components) == 2
        ## There is an estimator specification and a single model specification
        q_string = g_string = components[2]
    else
        q_string = g_string = "GLMNET"
    end
    models = TMLE.default_models(
        ## For the estimation of E[Y|W, T]: continuous outcome
        Q_continuous = model_from_string(string(q_string, "_REGRESSOR"), treatment_variables),
        ## For the estimation of E[Y|W, T]: binary outcome
        Q_binary = model_from_string(string(q_string, "_CLASSIFIER"), treatment_variables),
        ## For the estimation of p(T| W)
        G = model_from_string(string(g_string, "_CLASSIFIER"), treatment_variables; interactions=false),
    )
    # Create Estimators
    resampling = RESAMPLING(treatment_variables)
    estimators_strings = split(components[1], "-")
    estimators = [estimator_from_string(estimator_string, models, resampling, prevalence=prevalence, prevalence_file=prevalence_file) for estimator_string in estimators_strings]
    estimator_names = Tuple(Symbol(estimator_string, :_, q_string, :_, g_string) for estimator_string in estimators_strings)
    return NamedTuple{estimator_names}(estimators)
end