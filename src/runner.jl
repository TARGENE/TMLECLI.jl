instantiate_estimators(file::AbstractString) = load_tmle_spec(;file=file)
instantiate_estimators(estimators) = estimators

function load_tmle_spec(;file="glmnet")
    file = endswith(file, ".jl") ? file : joinpath(
        pkgdir(TargetedEstimation),
        "estimators-configs",
        string(file, ".jl"))
    include(abspath(file))
    return ESTIMATORS
end

mutable struct Runner
    estimators::NamedTuple
    estimands::Vector{TMLE.Estimand}
    dataset::DataFrame
    cache_manager::CacheManager
    chunksize::Int
    outputs::Outputs
    verbosity::Int
    failed_nuisance::Set
    function Runner(dataset; 
        estimands_config="factorialATE", 
        estimators_spec="glmnet",
        verbosity=0, 
        outputs=Outputs(), 
        chunksize=100,
        rng=123,
        cache_strategy="release-unusable",
        sort_estimands=false
        )    
        # Retrieve TMLE specifications
        estimators = instantiate_estimators(estimators_spec)
        # Load dataset
        dataset = instantiate_dataset(dataset)
        # Read parameter files
        estimands = instantiate_estimands(estimands_config, dataset)
        if sort_estimands
            estimands = groups_ordering(estimands; 
                brute_force=true, 
                do_shuffle=true, 
                rng=MersenneTwister(rng), 
                verbosity=verbosity
            )
        end
        cache_manager = make_cache_manager(estimands, cache_strategy)
        
        failed_nuisance = Set([])

        return new(estimators, estimands, dataset, cache_manager, chunksize, outputs, verbosity, failed_nuisance)
    end
end

function save(runner::Runner, results, partition, finalize)
    # Append JSON Output
    update_file(runner.outputs.json, results; finalize=finalize)
    # Append JLS Output
    update_file(runner.outputs.jls, results, runner.dataset)
    # Append HDF5 Output
    update_file(runner.outputs.hdf5, results, runner.dataset)
end

function try_estimation(runner, Ψ, estimator)
    try
        result, _ = estimator(Ψ, runner.dataset,
            cache=runner.cache_manager.cache,
            verbosity=runner.verbosity, 
        )
        return result
    catch e
        # Some nuisance function fits may fail. We do not interrupt on them but log instead.
        if e isa TMLE.FitFailedError
            # This also allows to skip fast the next estimands requiring the same nuisance functions.
            if !(e.model isa TMLE.Fluctuation)
                push!(runner.failed_nuisance, e.estimand)
            end
            return FailedEstimate(Ψ, e.msg)
        # On other errors, rethrow
        else 
            rethrow(e) 
        end
    end
end

function skip_fast(runner, Ψ)
    ηs = TMLE.nuisance_functions_iterator(Ψ)
    any(η ∈ runner.failed_nuisance for η in ηs) && return true
    return false
end

function (runner::Runner)(partition)
    results = Vector{NamedTuple}(undef, size(partition, 1))
    for (partition_index, param_index) in enumerate(partition)
        Ψ = runner.estimands[param_index]
        if skip_fast(runner, Ψ)
            results[partition_index] = NamedTuple{keys(runner.estimators)}([FailedEstimate(Ψ, "Skipped due to shared failed nuisance fit.") for _ in 1:length(runner.estimators)])
            continue
        end
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
end


"""
    tmle(dataset; 
        estimands="factorialATE", 
        estimators="glmnet"; 
        verbosity=0, 
        outputs=Outputs(),
        chunksize=100,
        rng=123,
        cache_strategy="release-unusable",
        sort_estimands=false
    )

TMLE CLI.

# Args

- `dataset`: Data file (either .csv or .arrow)

# Options

- `--estimands`: A string ("factorialATE") or a serialized TMLE.Configuration (accepted formats: .json | .yaml | .jls)
- `--estimators`: A julia file containing the estimators to use.
- `-v, --verbosity`: Verbosity level.
- `-o, --outputs`: Ouputs to be generated.
- `--chunksize`: Results are written in batches of size chunksize.
- `-r, --rng`: Random seed (Only used for estimands ordering at the moment).
- `-c, --cache-strategy`: Caching Strategy for the nuisance functions, any of ("release-unusable", "no-cache", "max-size").

# Flags

- `-s, --sort_estimands`: Sort estimands to minimize cache usage (A brute force approach will be used, resulting in exponentially long sorting time).
"""
function tmle(dataset::String;
    estimands::String="factorialATE", 
    estimators::String="glmnet",
    verbosity::Int=0, 
    outputs::Outputs=Outputs(),
    chunksize::Int=100,
    rng::Int=123,
    cache_strategy::String="release-unusable",
    sort_estimands::Bool=false
    )
    runner = Runner(dataset;
        estimands_config=estimands, 
        estimators_spec=estimators, 
        verbosity=verbosity, 
        outputs=outputs, 
        chunksize=chunksize,
        rng=rng,
        cache_strategy=cache_strategy,
        sort_estimands=sort_estimands
    )
    runner()
    verbosity >= 1 && @info "Done."
    return
end
