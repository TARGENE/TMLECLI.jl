struct FailedEstimation
    message::String
end

mutable struct Runner
    estimators::NamedTuple
    estimands::Vector{TMLE.Estimand}
    dataset::DataFrame
    cache_manager::CacheManager
    chunksize::Int
    pvalue_threshold::Float64
    output_ios::NamedTuple
    function Runner(parsed_args)
        datafile = parsed_args["dataset"]
        paramfile = parsed_args["estimands-config"]
        estimatorfile = parsed_args["estimators-config"]
        verbosity = parsed_args["verbosity"]
        csv_filename = parsed_args["csv-out"]
        hdf5_filename = parsed_args["hdf5-out"]
        pvalue_threshold = parsed_args["pval-threshold"]
        chunksize = parsed_args["chunksize"]
        rng = parsed_args["rng"]
        cache_strategy = parsed_args["cache-strategy"]
        sort_estimands = parsed_args["sort-estimands"]
    
        # Output IOs
        output_ios = (CSV=csv_filename, HDF5=hdf5_filename)
        # Retrieve TMLE specifications
        estimators = TargetedEstimation.load_tmle_spec(estimatorfile)
        # Load dataset
        dataset = TargetedEstimation.instantiate_dataset(datafile)
        # Read parameter files
        estimands = TargetedEstimation.proofread_estimands_from_yaml(paramfile, dataset)
        if sort_estimands
            estimands = groups_ordering(estimands; 
                brute_force=true, 
                do_shuffle=true, 
                rng=MersenneTwister(rng), 
                verbosity=verbosity
            )
        end
        cache_manager = make_cache_manager(estimands, cache_strategy)

        return new(estimators, estimands, dataset, cache_manager, chunksize, pvalue_threshold, output_ios)
    end
end


function (runner::Runner)(partition)
    results = Vector{NamedTuple}(undef, size(partition, 1))
    for (partition_index, param_index) in enumerate(partition)
        Ψ = runner.estimands[param_index]
        # Make sure data types are appropriate for the estimand
        TargetedEstimation.coerce_types!(runner.dataset, Ψ)
        # Maybe update cache with new η_spec
        estimators_results = []
        for estimator in estimators
            try
                result, _ = estimator(Ψ, runner.dataset,
                    cache=runner.cache,
                    verbosity=runner.verbosity, 
                )
            catch e
                # On Error, store the nuisance function where the error occured 
                # to fail fast the next estimands
                result = FailedEstimation(string(e))
            end
            push!(estimators_results, result)
        end
        # Update results
        results[partition_index] = NamedTuple{keys(runner.estimators)}(estimators_results)
        # Release cache
        release!(runner.cache_manager, Ψ)
        # Try clean C memory
        GC.gc()
        if Sys.islinux()
            ccall(:malloc_trim, Cvoid, (Cint,), 0)
        end
    end
    return results
end

function (runner::Runner)()
    # Split worklist in partitions
    nparams = size(runner.estimands, 1)
    for partition in Iterators.partition(1:nparams, runner.chunksize)
        results = runner(partition)
        # Append CSV result with partition
        append_csv(csv_file, results)
        # Append HDF5 result if save-ic is true
        update_jld2_output(jld2_file, partition, results, dataset; pval_threshold=pval_threshold)
    end

    verbosity >= 1 && @info "Done."
    return 0
end

run_estimation(parsed_args) = Runner(parsed_args)()