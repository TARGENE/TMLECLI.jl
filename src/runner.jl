function load_julia_estimators(file)
    include(abspath(file))
    return ESTIMATORS
end

"""
    parse_prevalence_range(prevalence_range::AbstractString)

Parse a prevalence range string "lower,upper" to a tuple (lower, upper).
"""
function parse_prevalence_range(prevalence_range::AbstractString)
    parts = split(prevalence_range, ",")
    if length(parts) != 2
        throw(ArgumentError("prevalence-range must be in the format 'lower,upper' (e.g., '0.006,0.013')"))
    end
    lower = parse(Float64, strip(parts[1]))
    upper = parse(Float64, strip(parts[2]))
    if lower >= upper
        throw(ArgumentError("Lower bound of prevalence range must be less than upper bound"))
    end
    if lower <= 0 || upper >= 1
        throw(ArgumentError("Prevalence bounds must be between 0 and 1 (exclusive)"))
    end
    return (lower, upper)
end

parse_prevalence_range(::Nothing) = nothing

"""
    get_prevalence_grid(prevalence, prevalence_range, n_points)

Construct a grid of prevalence values for sensitivity analysis.

- If `prevalence_range` is nothing, return a single-element vector with `prevalence` (or nothing)
- If `prevalence_range` is provided:
  - Throw ArgumentError if `prevalence` is provided but outside the range
  - Use `n_points` evenly spaced values within the range including the bounds
  - Return sorted unique values
"""
function get_prevalence_grid(prevalence, prevalence_range::Nothing, n_points)
    return [prevalence]
end

function get_prevalence_grid(prevalence, prevalence_range::Tuple{Float64, Float64}, n_points)
    lower, upper = prevalence_range
    
    if !isnothing(prevalence) && (prevalence < lower || prevalence > upper)
        throw(ArgumentError("Prevalence $prevalence is outside the specified range [$lower, $upper]"))
    end
    
    # Create evenly spaced grid
    grid = collect(range(lower, upper, length=n_points))
    
    # Add prevalence if provided and not already in grid
    if !isnothing(prevalence)
        push!(grid, prevalence)
    end
    
    return sort(unique(grid))
end

"""
If estimators is an AbstractString, it is either:

- A Julia file containing a `ESTIMATORS` NamedTuple of these
estimators
- A string corresponding to estimators that can be constructed from the registry.
"""
function instantiate_estimators(config::AbstractString, estimands; prevalence=nothing)
    if endswith(config, ".jl")
        if !isnothing(prevalence)
            @error "Prevalence is not used when loading estimators from a file. You must specify it in the estimators themselves."
        end
        load_julia_estimators(config)
    else
        treatment_variables = treatments_from_estimands(estimands)
        estimators_from_string(config_string=config, treatment_variables=treatment_variables, prevalence=prevalence)
    end
end

"""
If estimators is something else than an AbstractString, it is simply assumed to be a properly formed 
NamedTuple of estimators.
"""
instantiate_estimators(estimators, estimands; prevalence=nothing) = estimators

mutable struct Runner
    estimators::NamedTuple
    estimands::Vector{TMLE.Estimand}
    dataset::DataFrame
    cache_manager::CacheManager
    chunksize::Int
    outputs::Outputs
    verbosity::Int
    failed_nuisance::Set
    save_sample_ids::Bool
    pvalue_threshold::Union{Nothing, Float64}
    prevalence::Union{Nothing, Float64}
    function Runner(dataset; 
        estimands_config="factorialATE", 
        estimators_spec="glmnet",
        verbosity=0, 
        outputs=Outputs(), 
        chunksize=100,
        rng=123,
        cache_strategy="release-unusable",
        sort_estimands=false,
        save_sample_ids=false,
        pvalue_threshold=nothing,
        prevalence=nothing
        )    
        # Load dataset
        dataset = instantiate_dataset(dataset)
        # Read parameter files
        estimands = instantiate_estimands(estimands_config, dataset)
        # Retrieve TMLE specifications
        estimators = instantiate_estimators(estimators_spec, estimands, prevalence=prevalence)
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

        return new(
            estimators, 
            estimands, 
            dataset, 
            cache_manager, 
            chunksize, 
            outputs, 
            verbosity, 
            failed_nuisance, 
            save_sample_ids, 
            pvalue_threshold,
            prevalence
        )
    end
end

function update_outputs(runner::Runner, results)
    results = runner.save_sample_ids ?
        add_sample_ids_to_results(results, runner.dataset) :
        results
    update(runner.outputs::Outputs, results)
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
            push!(runner.failed_nuisance, e.estimand)
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
            skipped_result = NamedTuple{keys(runner.estimators)}([FailedEstimate(Ψ, "Skipped due to shared failed nuisance fit.") for _ in 1:length(runner.estimators)])
            results[partition_index] = isnothing(runner.prevalence) ? skipped_result : merge(skipped_result, (PREVALENCE = runner.prevalence,))
            continue
        end
        # Make sure data types are appropriate for the estimand
        TMLECLI.coerce_types!(runner.dataset, Ψ)
        # Maybe update cache with new η_spec
        estimators_results = []
        for estimator in runner.estimators
            result = try_estimation(runner, Ψ, estimator)
            push!(
                estimators_results, 
                TMLE.emptyIC(result, runner.pvalue_threshold)
            )
        end
        # Update results (add PREVALENCE field only for CCW-TMLE, i.e., when prevalence is specified)
        estimator_results_nt = NamedTuple{keys(runner.estimators)}(estimators_results)
        results[partition_index] = isnothing(runner.prevalence) ? estimator_results_nt : merge(estimator_results_nt, (PREVALENCE = runner.prevalence,))
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

function (runner::Runner)(;skip_init_finalize::Bool=false)
    # Initialize output files
    skip_init_finalize || initialize(runner.outputs)
    # Run and update output files in batches
    nparams = size(runner.estimands, 1)
    for partition in Iterators.partition(1:nparams, runner.chunksize)
        results = runner(partition)
        update_outputs(runner, results)
    end
    # Finalize output files
    skip_init_finalize || finalize(runner.outputs)
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
- `--prevalence`: If the true prevalence of the outcome is known in the population, it can be specified here to correct for sampling bias.
- `--prevalence-range`: Range of prevalence values for sensitivity analysis (e.g., "0.006,0.013").
- `--n-prevalence-points`: Number of prevalence points within the range (default: 5).

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
    sort_estimands::Bool=false,
    save_sample_ids=false,
    pvalue_threshold=nothing,
    prevalence=nothing,
    prevalence_range=nothing,
    n_prevalence_points::Int=5
    )
    # Parse prevalence range if provided
    parsed_range = parse_prevalence_range(prevalence_range)
    
    # Get prevalence grid for sensitivity analysis
    prevalence_grid = get_prevalence_grid(prevalence, parsed_range, n_prevalence_points)
    
    verbosity >= 1 && !isnothing(parsed_range) && @info "Running sensitivity analysis over $(length(prevalence_grid)) prevalence values: $prevalence_grid"
    
    initialize(outputs)
    # If no prevalence is provided standard tmle is ran 
    # If prevalence is specifed without range it will also run once with ccw-tmle
    for (i, prev) in enumerate(prevalence_grid)
        verbosity >= 1 && !isnothing(parsed_range) && @info "Estimating with prevalence = $prev ($(i)/$(length(prevalence_grid)))"
        
        runner = Runner(dataset;
            estimands_config=estimands, 
            estimators_spec=estimators, 
            verbosity=verbosity, 
            outputs=outputs, 
            chunksize=chunksize,
            rng=rng,
            cache_strategy=cache_strategy,
            sort_estimands=sort_estimands,
            save_sample_ids=save_sample_ids,
            pvalue_threshold=pvalue_threshold,
            prevalence=prev
        )
        runner(;skip_init_finalize=true)
    end
    
    finalize(outputs)
    
    verbosity >= 1 && @info "Done."
    return
end
