asarrayed(val::AbstractArray) = val
asarrayed(val) = [val]


function buildmodels(config)
    models = Dict()
    for model_spec in config
        modelname = pop!(model_spec, :type)
        modeltype = eval(Symbol(modelname))
        paramnames = Tuple(keys(model_spec))
        counter = 1
        for paramvals in Base.Iterators.product((asarrayed(x) for x in values(model_spec))...)
            model = modeltype(;NamedTuple{paramnames}(paramvals)...)
            models[Symbol(modelname*"_$counter")] = model
            counter += 1
        end
    end
    return models
end

function resampling_from_config(resampling_info)
    resampling_type = eval(Symbol(resampling_info[:type]))
    resampling = resampling_type()
    if haskey(resampling_info, :nfolds)
        resampling = resampling_type(nfolds=resampling_info[:nfolds])
    end

    if haskey(resampling_info, :adaptive)
        resampling = AdaptiveCV(resampling)
    end

    return resampling
end

function stack_from_config(config::Dict, metalearner)
    # Define the resampling strategy
    resampling = CV()
    if haskey(config, :resampling)
        resampling = resampling_from_config(config[:resampling])
    end

    # Define the internal cross validation measures to report
    measures = (haskey(config, :measures) && size(config[:measures], 1) > 0) ? 
                    [getfield(MLJ, Symbol(fn)) for fn in config[:measures]] : 
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

function tmle_spec_from_yaml(yamlfile)
    config = YAML.load_file(yamlfile; dicttype=Dict{Symbol,Any})

    # Build G estimator
    if config[:G][:model] == "Stack"
        G = stack_from_config(config[:G], LogisticClassifier(fit_intercept=false, lambda=0))
    else
        G = learner_from_config(config[:G])
    end


    if config[:Q_continuous][:model] == "Stack"
        metalearner =  LinearRegressor(fit_intercept=false)
        Q_continuous = stack_from_config(config[:Q_continuous], metalearner)
    else
        Q_continuous = learner_from_config(config[:Q_continuous])
    end

    if config[:Q_binary][:model] == "Stack"
        metalearner =  LogisticClassifier(fit_intercept=false, lambda=0)
        Q_binary = stack_from_config(config[:Q_binary], metalearner)
    else
        Q_binary = learner_from_config(config[:Q_binary])
    end

    threshold = haskey(config, :threshold) ? config[:threshold] : 1e-8

    return (G=G, Q_continuous=Q_continuous, Q_binary=Q_binary, threshold=threshold)
end