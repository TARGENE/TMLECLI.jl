###############################################################################
# BUILD TMLE FROM .TOML

function buildmodels(config)
    models = Dict()
    for model_spec in config
        modelname = pop!(model_spec, :type)
        modeltype = eval(Symbol(modelname))
        paramnames = Tuple(keys(model_spec))
        counter = 1
        for paramvals in Base.Iterators.product(values(model_spec)...)
            model = modeltype(;NamedTuple{paramnames}(paramvals)...)
            models[Symbol(modelname*"_$counter")] = model
            counter += 1
        end
    end
    return models
end


function stack_from_config(config::Dict, metalearner)
    # Define the resampling strategy
    resampling = CV()
    if haskey(config, :resampling)
        resampling_info = config[:resampling]
        nfolds = haskey(resampling_info, :nfolds) ? resampling_info[:nfolds] : resampling.nfolds
        resampling = eval(Symbol(resampling_info[:type]))(nfolds=nfolds)
        if haskey(resampling_info, :adaptive)
            resampling = AdaptiveCV(resampling)
        end
    end

    # Define the internal cross validation measures to report
    measures = (haskey(config, :measures) && size(config[:measures], 1) > 0) ? 
                    [getfield(MLJBase, Symbol(fn)) for fn in config[:measures]] : 
                    nothing
        
    # Define the models library
    models = buildmodels(config[:models])

    # Define the Stack
    Stack(;metalearner=metalearner, resampling=resampling, measures=measures, cache=false, models...)
end

function learner_from_config(config)
    modeltype = eval(Symbol(pop!(config, :model)))
    return modeltype(;config...)
end

function tmle_from_yaml(yamlfile, queries, target_type)
    config = YAML.load_file(yamlfile; dicttype=Dict{Symbol,Any})

    # Build G estimator
    if config[:G][:model] == "Stack"
        G = stack_from_config(config[:G], LogisticClassifier(fit_intercept=false))
    else
        G = learner_from_config(config[:G])
    end
    if length(first(queries).case) > 1
        G = FullCategoricalJoint(G)
    end

    # Build Q estimator
    if target_type == Real
        if config[:Q_continuous][:model] == "Stack"
            metalearner =  LinearRegressor(fit_intercept=false)
            Q = stack_from_config(config[:Q_continuous], metalearner)
        else
            Q = learner_from_config(config[:Q_continuous])
        end
    elseif target_type == Bool
        if config[:Q_binary][:model] == "Stack"
            metalearner =  LogisticClassifier(fit_intercept=false)
            Q = stack_from_config(config[:Q_binary], metalearner)
        else
            Q = learner_from_config(config[:Q_binary])
        end
    else
        throw(ArgumentError("The type of the outcomes: $target_type, should be either a Float or a Bool"))
    end

    threshold = haskey(config, :threshold) ? config[:threshold] : 0.005

    return TMLEstimator(Q, G, queries...; threshold=threshold)
end