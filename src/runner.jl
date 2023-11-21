struct FailedEstimation
    message::String
end

@option struct JSONOutput
    filename::Union{Nothing, String} = nothing
    pval_threshold::Union{Nothing, Float64} = nothing
end

initialize(output::JSONOutput) = initialize_json(output.filename)

@option struct HDF5Output
    filename::Union{Nothing, String} = nothing
    pval_threshold::Union{Nothing, Float64} = nothing
end

@option struct Outputs
    json::JSONOutput = JSONOutput()
    hdf5::HDF5Output = HDF5Output()
    std::Bool = false
end

function initialize(outputs::Outputs)
    initialize(outputs.json)
end

mutable struct Runner
    estimators::NamedTuple
    estimands::Vector{TMLE.Estimand}
    dataset::DataFrame
    cache_manager::CacheManager
    chunksize::Int
    outputs::Outputs
    verbosity::Int
    function Runner(dataset, estimands, estimators; 
        verbosity=0, 
        outputs=Outputs(), 
        chunksize=100,
        rng=123,
        cache_strategy="release-unusable",
        sort_estimands=false
        )    
        # Retrieve TMLE specifications
        estimators = TargetedEstimation.load_tmle_spec(estimators)
        # Load dataset
        dataset = TargetedEstimation.instantiate_dataset(dataset)
        # Read parameter files
        estimands = TargetedEstimation.proofread_estimands(estimands, dataset)
        if sort_estimands
            estimands = groups_ordering(estimands; 
                brute_force=true, 
                do_shuffle=true, 
                rng=MersenneTwister(rng), 
                verbosity=verbosity
            )
        end
        cache_manager = make_cache_manager(estimands, cache_strategy)

        return new(estimators, estimands, dataset, cache_manager, chunksize, outputs, verbosity)
    end
end

function save(runner::Runner, results, partition, finalize)
    # Append STD Out
    update(runner.outputs.std, results, partition)
    # Append JSON Output
    update(runner.outputs.json, results; finalize=finalize)
    # Append HDF5 Output
    update(runner.outputs.hdf5, partition, results, runner.dataset)
end


function try_estimation(runner, Ψ, estimator)
    try
        result, _ = estimator(Ψ, runner.dataset,
            cache=runner.cache_manager.cache,
            verbosity=runner.verbosity, 
        )
        return result
    catch e
        # On Error, store the nuisance function where the error occured 
        # to fail fast the next estimands
        return FailedEstimation(string(e))
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
        for estimator in runner.estimators
            result = try_estimation(runner, Ψ, estimator)
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
    # Initialize output files
    initialize(runner.outputs)
    # Split worklist in partitions
    nparams = size(runner.estimands, 1)
    partitions = collect(Iterators.partition(1:nparams, runner.chunksize))
    for partition in partitions
        results = runner(partition)
        save(runner, results, partition, partition===partitions[end])
    end
    runner.verbosity >= 1 && @info "Done."
    return 0
end


"""
TMLE CLI.

# Args

- `dataset`: Data file (either .csv or .arrow)
- `estimands`: Estimands file (either .json or .yaml)
- `estimators`: A julia file containing the estimators to use.

# Options

- `-v, --verbosity`: Verbosity level.
- `-j, --json_out`: JSON output filename.
- `--hdf5_out`: HDF5 output filename.
- `--chunksize`: Results are written in batches of size chunksize.
- `-r, --rng`: Random seed (Only used for estimands ordering at the moment).
- `-c, --cache_strategy`: Caching Strategy for the nuisance functions, any of ("release-unusable", "no-cache", "max-size").

# Flags

- `-s, --sort_estimands`: Sort estimands to minimize cache usage (A brute force approach will be used, resulting in exponentially long sorting time).
"""
@main function tmle(dataset, estimands, estimators; 
    verbosity=0, 
    outputs=Outputs(),
    chunksize=100,
    rng=123,
    cache_strategy="release-unusable",
    sort_estimands=false
    )
    runner = Runner(dataset, estimands, estimators; 
        verbosity=verbosity, 
        outputs=outputs, 
        chunksize=chunksize,
        rng=rng,
        cache_strategy=cache_strategy,
        sort_estimands=sort_estimands
    )
    runner()
    return
end