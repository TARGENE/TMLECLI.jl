function update_model_or_ranges!(model, ranges, key, val::String)
    string_vals = split(replace(val, " " => ""), ",")
    lower = parse(Float64, string_vals[1])
    upper = parse(Float64, string_vals[2])
    scale = size(string_vals, 1) > 2 ? Symbol(string_vals[3]) : nothing
    push!(
        ranges,
        range(model, Symbol(key), lower=lower, upper=upper, scale=scale)
    )
end

update_model_or_ranges!(model, ranges, key, val) = setproperty!(model, Symbol(key), val)

GridSearchEvoTreeRegressor(;kwargs...) = GridSearchModel(EvoTreeRegressor, rmse; kwargs...)
GridSearchEvoTreeClassifier(;kwargs...) = GridSearchModel(EvoTreeClassifier, log_loss; kwargs...)

function GridSearchModel(model_class, measure; resolution=10, resampling=nothing, kwargs...)
    ranges = []
    resampling = resampling === nothing ? Holdout() : resampling_from_config(resampling)
    model = model_class()
    for (key, val) in kwargs
        update_model_or_ranges!(model, ranges, key, val)
    end
    return TunedModel(
        model=model,
        resampling=resampling,
        tuning=Grid(resolution=resolution),
        range=ranges,
        measure=measure
        )
end
