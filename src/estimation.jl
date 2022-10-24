###############################################################################
####################           Main Function               ####################
###############################################################################

function try_tmle_run!(cache, Ψ, η_spec, dataset; verbosity=1, threshold=1e-8)
    try
        tmle_result, initial_result, cache = tmle_run!(cache, Ψ, η_spec, dataset; verbosity=verbosity, threshold=threshold)
        return tmle_result, initial_result, missing
    catch e
        return missing, missing, string(e)
    end
end


function main(parsed_args)
    verbosity = parsed_args["verbosity"]
    save_full = SaveFull{parsed_args["save-full"]}()
    outfile = parsed_args["out"]
    parameters = TMLE.parameters_from_yaml(parsed_args["param-file"])
    non_target_columns = TargetedEstimation.get_non_target_columns(first(parameters))
    parameters_df = DataFrame(TARGET=[p.target for p in parameters], PARAMETER=parameters)

    dataset = TargetedEstimation.instantiate_dataset(parsed_args["data"])
    TargetedEstimation.make_categorical!(dataset, keys(first(parameters).treatment), true)

    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(parsed_args["estimator-file"])

    cache = nothing
    open_io(outfile, save_full) do io
        gpd_parameters = groupby(parameters_df, :TARGET)
        initialize_outfile(io, gpd_parameters)
        for (target_key, target_parameters) in pairs(gpd_parameters)
            target = target_key.TARGET
            verbosity >= 1 && @info string("Targeted Estimation for target: ", target)
            targetisbinary = TargetedEstimation.isbinarytarget(dataset[!, target])
            TargetedEstimation.make_categorical!(dataset, target, targetisbinary)
            η_spec = TargetedEstimation.nuisance_spec_from_target(tmle_spec, targetisbinary)
            n_params = size(target_parameters[!, :PARAMETER], 1)
            tmle_results = Vector{Union{TMLE.AbstractTMLE, Missing}}(undef, n_params)
            initial_estimates = Vector{Union{Float64, Missing}}(undef, n_params)
            logs = Vector{Union{String, Missing}}(undef, n_params)
            for param_index in 1:n_params
                Ψ = target_parameters[param_index, :PARAMETER]
                tmle_result, initial_result, log = try_tmle_run!(cache, Ψ, η_spec, dataset; verbosity=verbosity, threshold=tmle_spec.threshold)
                tmle_results[param_index] = tmle_result
                initial_estimates[param_index] = TMLE.estimate(initial_result)
                logs[param_index] = log
            end
            sample_ids = TargetedEstimation.get_sample_ids(dataset, vcat(target, non_target_columns), save_full)
            write_target_results(io, target_parameters, tmle_results, initial_estimates, sample_ids, logs)
        end
        
    end

    verbosity >= 1 && @info "Done."
    return 0
end
