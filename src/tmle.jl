###############################################################################
####################           Main Function               ####################
###############################################################################


function tmle_estimation(parsed_args)
    verbosity = parsed_args["verbosity"]
    outprefix = parsed_args["outprefix"]
    pval_threshold = parsed_args["pval-threshold"]
    save_ic = parsed_args["save-ic"]
    # Load dataset
    dataset = TargetedEstimation.instantiate_dataset(parsed_args["data"])
    # Read parameter files
    parameters = parameters_from_yaml(parsed_args["param-file"], dataset)
    parameters_df = DataFrame(TARGET=[p.target for p in parameters], PARAMETER=parameters)
    # Get covariate, confounder and treatment columns
    treatment_cols = keys(first(parameters).treatment)
    covariate_cols = first(parameters).covariates
    confounder_cols = first(parameters).confounders
    non_target_columns = TargetedEstimation.get_non_target_columns(treatment_cols, covariate_cols, confounder_cols)

    # Treatment columns are converted to categorical
    TargetedEstimation.make_categorical!(dataset, treatment_cols, infer_ordered=true)
    
    # Confounders and Covariates are converted to Float64
    TargetedEstimation.make_float!(dataset, vcat(covariate_cols, confounder_cols))
    
    # Retrieve TMLE specifications
    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(parsed_args["estimator-file"])

    at_least_one_sieve = false
    cache = TMLECache(dataset)
    gpd_parameters = groupby(parameters_df, :TARGET)
    csv_io = initialize_csv_io(outprefix)
    jld_io = initialize_jld_io(outprefix, gpd_parameters, save_ic)
    for (target_key, target_parameters) in pairs(gpd_parameters)
        target = target_key.TARGET
        verbosity >= 1 && @info string("Targeted Estimation for target: ", target)
        targetisbinary = TargetedEstimation.isbinarytarget(dataset[!, target])
        targetisbinary && make_categorical!(dataset, target)
        η_spec = TargetedEstimation.nuisance_spec_from_target(tmle_spec, targetisbinary, tmle_spec.cache)
        n_params = size(target_parameters[!, :PARAMETER], 1)
        tmle_results = Vector{Union{TMLE.TMLEResult, Missing}}(undef, n_params)
        logs = Vector{Union{String, Missing}}(undef, n_params)
        for param_index in 1:n_params
            Ψ = target_parameters[param_index, :PARAMETER]
            tmle_result, log = try_tmle!(cache, Ψ, η_spec; verbosity=verbosity, threshold=tmle_spec.threshold)
            tmle_results[param_index] = tmle_result
            logs[param_index] = log
        end
        # Append CSV result for target
        append_csv(csv_io, target_parameters, tmle_results, logs)
        # Append HDF5 result if save-ic is true
        if parsed_args["save-ic"]
            # Append only with results passing the threshold
            mask = [i for i in 1:n_params if (tmle_results[i] !== missing) && (pvalue(OneSampleZTest(tmle_results[i].tmle)) <= pval_threshold)]
            if size(mask, 1) > 0
                at_least_one_sieve = true
                sample_ids = TargetedEstimation.get_sample_ids(dataset, vcat(target, non_target_columns))
                append_hdf5(jld_io, no_slash(target), tmle_results, logs, sample_ids, mask)
            end
        end
        # See if this helps with memory issue
        GC.gc()
        if Sys.islinux()
            ccall(:malloc_trim, Cvoid, (Cint,), 0)
        end
    end
    
    # Close io files
    if jld_io !== nothing
        if !at_least_one_sieve
            rm(string(outprefix, ".hdf5"))
        end
    end

    verbosity >= 1 && @info "Done."
    return 0
end
