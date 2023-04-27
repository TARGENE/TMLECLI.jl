struct MissingTMLEResult
    parameter::TMLE.Parameter
end

function try_tmle!(cache; verbosity=1, threshold=1e-8)
    try
        tmle_result, _ = tmle!(cache; verbosity=verbosity, threshold=threshold)
        return tmle_result, missing
    catch e
        @warn string("Failed to run Targeted Estimation for parameter:", cache.Ψ)
        return MissingTMLEResult(cache.Ψ), string(e)
    end
end


function partition_tmle!(
    cache, 
    tmle_results, 
    logs, 
    partition,
    tmle_spec,
    parameters,
    variables; 
    verbosity=0)
    for (partition_index, param_index) in enumerate(partition)
        previous_target_is_binary = isdefined(cache, :Ψ) ? cache.Ψ.target ∈ variables.binarytargets : nothing
        Ψ = parameters[param_index]
        # Update cache with new Ψ
        update!(cache, Ψ)
        # Maybe update cache with new η_spec
        target_is_binary = Ψ.target ∈ variables.binarytargets
        if !isdefined(cache, :η_spec) || !(target_is_binary === previous_target_is_binary)
            Q_spec = target_is_binary ? tmle_spec.Q_binary : tmle_spec.Q_continuous
            η_spec = NuisanceSpec(Q_spec, tmle_spec.G, cache=tmle_spec.cache)
            update!(cache, η_spec)
        end
        # Run TMLE
        tmle_result, log = TargetedEstimation.try_tmle!(cache; verbosity=verbosity, threshold=tmle_spec.threshold)
        # Update results
        tmle_results[partition_index] = tmle_result
        logs[partition_index] = log
    end
end

function tmle_estimation(parsed_args)
    datafile = parsed_args["data"]
    paramfile = parsed_args["param-file"]
    estimatorfile = parsed_args["estimator-file"]
    verbosity = parsed_args["verbosity"]
    csv_file = parsed_args["csv-out"]
    jld2_file = parsed_args["hdf5-out"]
    pval_threshold = parsed_args["pval-threshold"]
    chunksize = parsed_args["chunksize"]

    # Load dataset
    dataset = TargetedEstimation.instantiate_dataset(datafile)
    # Read parameter files
    parameters = TargetedEstimation.read_parameters(paramfile, dataset)
    optimize_ordering!(parameters)

    # Get covariate, confounder and treatment columns
    variables = TargetedEstimation.variables(parameters, dataset)
    TargetedEstimation.coerce_types!(dataset, variables)
    
    # Retrieve TMLE specifications
    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(estimatorfile)

    cache = TMLECache(dataset)
    nparams = size(parameters, 1)
    for partition in Iterators.partition(1:nparams, chunksize)
        partition_size = size(partition, 1)
        tmle_results = Vector{Union{TMLE.TMLEResult, MissingTMLEResult}}(undef, partition_size)
        logs = Vector{Union{String, Missing}}(undef, partition_size)
        partition_tmle!(cache, tmle_results, logs, partition, tmle_spec, parameters, variables; verbosity=verbosity)
        # Append CSV result with partition
        append_csv(csv_file, tmle_results, logs)
        # Append HDF5 result if save-ic is true
        update_jld2_output(jld2_file, partition, tmle_results, dataset; pval_threshold=pval_threshold)
    end

    verbosity >= 1 && @info "Done."
    return 0
end
