
###############################################################################
####################             Utilities                 ####################
###############################################################################

get_sample_ids(data, targets_columns) = dropmissing(data[!, [:SAMPLE_ID, targets_columns...]]).SAMPLE_ID


"""
    instantiate_dataset(path::String)

Returns a DataFrame wrapper around a dataset, either in CSV or Arrow format.
"""
function instantiate_dataset(path::String)
    if endswith(path, "arrow")
        return DataFrame(Arrow.Table(path))
    else
        return CSV.read(path, DataFrame)
    end
end

isbinarytarget(y::AbstractVector) = Set(unique(skipmissing(y))) == Set([0, 1])

function nuisance_spec_from_target(tmle_spec, isbinary)
    Q_spec = isbinary ? tmle_spec.Q_binary : tmle_spec.Q_continuous
    return NuisanceSpec(Q_spec, tmle_spec.G)
end

maybe_categorical(v) = categorical(v)
maybe_categorical(v::CategoricalArray) = v

make_categorical!(dataset, colname::Union{String, Symbol}, isbinary::Bool) =
    isbinary ? dataset[!, colname] = maybe_categorical(dataset[!, colname]) : nothing

function make_categorical!(dataset, colnames::Tuple, isbinary::Bool)
    for colname in colnames
        make_categorical!(dataset, colname, isbinary)
    end
end

function save_estimation_results(target_group, tmle_results, initial_estimates, sample_ids)
    target_group["tmle_results"] = tmle_results
    target_group["initial_estimates"] = initial_estimates
    target_group["sample_ids"] = sample_ids
end

get_non_target_columns(parameter) =
    vcat(keys(parameter.treatment)..., parameter.confounders, parameter.covariates)

###############################################################################
####################           Main Function               ####################
###############################################################################

function tmle_run(parsed_args)
    verbosity = parsed_args["verbosity"]
    parameters = TMLE.parameters_from_yaml(parsed_args["param-file"])
    non_target_columns = get_non_target_columns(first(parameters))
    parameters_df = DataFrame(TARGET=[p.target for p in parameters], PARAMETER=parameters)

    dataset = TargetedEstimation.instantiate_dataset(parsed_args["data"])

    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(parsed_args["estimator-file"])

    cache = nothing
    jldopen(parsed_args["out"], "w", compress=true) do io
        results_group = JLD2.Group(io, "results")
        gpd_parameters = groupby(parameters_df, :TARGET)
        for (target_key, target_parameters) in pairs(gpd_parameters)
            target = target_key.TARGET
            verbosity >= 1 && @info string("Targeted Estimation for target: ", target)
            target_group = JLD2.Group(results_group, string(target))
            targetisbinary = TargetedEstimation.isbinarytarget(dataset[!, target])
            TargetedEstimation.make_categorical!(dataset, target, targetisbinary)
            η_spec = TargetedEstimation.nuisance_spec_from_target(tmle_spec, targetisbinary)
            tmle_results = TMLE.AbstractTMLE[]
            initial_estimates = Float64[]
            for Ψ in target_parameters[!, :PARAMETER]
                TargetedEstimation.make_categorical!(dataset, keys(Ψ.treatment), true)
                if cache === nothing
                    tmle_result, initial_result, cache = tmle(Ψ, η_spec, dataset; verbosity=verbosity-1, threshold=tmle_spec.threshold)
                else
                    tmle_result, initial_result, cache = tmle!(cache; verbosity=verbosity-1, threshold=tmle_spec.threshold)
                end
                push!(tmle_results, tmle_result)
                push!(initial_estimates, estimate(initial_result))
            end
            sample_ids = get_sample_ids(dataset, vcat(target, non_target_columns))
            save_estimation_results(target_group, tmle_results, initial_estimates, sample_ids)
            # push!(io["sample_ids"], sample_ids)
        end
        io["parameters"] = first(gpd_parameters)[!, "PARAMETER"]
    end

    verbosity >= 1 && @info "Done."
    return 0
end
