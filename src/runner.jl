function load_julia_estimators(file)
    include(abspath(file))
    return ESTIMATORS
end

"""
If estimators is an AbstractString, it is either:

- A Julia file containing a `ESTIMATORS` NamedTuple of these
estimators
- A string corresponding to estimators that can be constructed from the registry.
"""
function instantiate_estimators(config::AbstractString, estimands; prevalence_mode="sampling", prevalence_map=nothing)
    if endswith(config, ".jl")
        if !isnothing(prevalence_map)
            @error "Prevalence is not used when loading estimators from a file. You must specify it in the estimators themselves."
        end
        load_julia_estimators(config)
    else
        treatment_variables = treatments_from_estimands(estimands)
        estimator_prevalence_arg = prevalence_mode == "ccw" && prevalence_map !== nothing ? prevalence_map : nothing
        estimators_from_string(config_string=config, treatment_variables=treatment_variables, prevalence=estimator_prevalence_arg)
    end
end

"""
If estimators is something else than an AbstractString, it is simply assumed to be a properly formed 
NamedTuple of estimators.
"""
instantiate_estimators(estimators, estimands; prevalence_mode="sampling", prevalence_map=nothing) = estimators

instantiate_prevalence_map(::Nothing) = nothing

function instantiate_prevalence_map(prevalence_file)
    prevalence_table = CSV.read(prevalence_file, DataFrame; header=[:OUTCOME, :PREVALENCE], delim="\t")
    return Dict(zip(prevalence_table.OUTCOME, prevalence_table.PREVALENCE))
end

mutable struct Runner
    estimators::NamedTuple
    estimands::Vector{TMLE.Estimand}
    dataset::DataFrame
    cache_manager::CacheManager
    chunksize::Int
    rng_seed::Int
    outputs::Outputs
    verbosity::Int
    failed_nuisance::Set
    save_sample_ids::Bool
    pvalue_threshold::Union{Nothing, Float64}
    prevalence_map::Union{Nothing, Dict{String, Float64}}
    prevalence_mode::String
    output_downsampled_datasets::Bool

    function Runner(dataset; 
        estimands_config="factorialATE", 
        estimators_spec="glmnet",
        verbosity=0, 
        outputs=Outputs(), 
        chunksize=100,
        rng_seed=123,
        cache_strategy="release-unusable",
        sort_estimands=false,
        save_sample_ids=false,
        pvalue_threshold=nothing,
        prevalence_file=nothing,
        prevalence_mode="sampling",
	output_downsampled_datasets=true
        )    
        # Load dataset
        dataset = instantiate_dataset(dataset)
        # Read parameter files
        estimands = instantiate_estimands(estimands_config, dataset)
        # Load prevalence map
        prevalence_map = instantiate_prevalence_map(prevalence_file)
        # Retrieve TMLE specifications
        estimators = instantiate_estimators(estimators_spec, estimands, prevalence_mode=prevalence_mode, prevalence_map=prevalence_map)
        if sort_estimands
            estimands = groups_ordering(estimands; 
                brute_force=true, 
                do_shuffle=true, 
		rng=MersenneTwister(rng_seed), 
                verbosity=verbosity
            )
        end
        cache_manager = make_cache_manager(estimands, cache_strategy; prevalence_mode=prevalence_mode)
        
        failed_nuisance = Set([])

        return new(
            estimators, 
            estimands, 
            dataset, 
            cache_manager, 
            chunksize, 
            rng_seed,
	    outputs, 
            verbosity, 
            failed_nuisance, 
            save_sample_ids, 
            pvalue_threshold,
            prevalence_map,
            prevalence_mode,
	    output_downsampled_datasets
        )
    end
end

function update_outputs(runner::Runner, results)
    results = runner.save_sample_ids ?
        add_sample_ids_to_results(results, runner.dataset) :
        results
    update(runner.outputs::Outputs, results)
end

function try_estimation(runner, Ψ, estimator, dataset)
    dataset = if runner.prevalence_map !== nothing && runner.prevalence_mode == "sampling"
	downsample_dataset(runner.dataset, runner.prevalence_map, Ψ; rng_seed=runner.rng_seed)
    else
        runner.dataset
    end

    # Trait has zero prevalence in the dataset
    if dataset === nothing
        return FailedEstimate(
            Ψ,
            "Trait has a prevalence of zero and therefore cannot be prevalence-matched."
        )
    end

    try
        result, _ = estimator(
            Ψ,
            dataset;
            cache=runner.cache_manager.cache,
            verbosity=runner.verbosity,
        )
        return result
    catch e
        if e isa TMLE.FitFailedError
            push!(runner.failed_nuisance, e.estimand)
            return FailedEstimate(Ψ, e.msg)
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
        TMLECLI.coerce_types!(runner.dataset, Ψ)

	# downsampled dataset once for this Ψ
	dataset = downsample_and_write_dataset(
            runner,
            Ψ,
            param_index
        )
	if dataset === nothing
            results[partition_index] =
                NamedTuple{keys(runner.estimators)}(
                    [FailedEstimate(
                        Ψ,
                        "Trait has a prevalence of zero and therefore cannot be prevalence-matched."
                    ) for _ in runner.estimators]
                )
            continue
        end
	
        # Maybe update cache with new η_spec
        estimators_results = []
        for estimator in runner.estimators
	    result = try_estimation(runner, Ψ, estimator, dataset)
            push!(
                estimators_results, 
                TMLE.emptyIC(result, runner.pvalue_threshold)
            )
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
    # Run and update output files in batches
    nparams = size(runner.estimands, 1)
    for partition in Iterators.partition(1:nparams, runner.chunksize)
        results = runner(partition)
        update_outputs(runner, results)
    end
    # Finalize output files
    finalize(runner.outputs)
end


"""
    tmle(dataset; 
        estimands="factorialATE", 
        estimators="glmnet"; 
        verbosity=0, 
        outputs=Outputs(),
        chunksize=100,
        rng_seed=123,
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
- `-r, --rng_seed`: Random seed (Used for estimands ordering and prevalence-matched downsampling).
- `-c, --cache-strategy`: Caching Strategy for the nuisance functions, any of ("release-unusable", "no-cache", "max-size").
- `--prevalence`: If the true prevalence of the outcome is known in the population, it can be specified here to correct for sampling bias.
# Flags
- `--output_downsampled_datasets`: if using prevalence-matched downsampleing, specify whether you want to output the downsampled datasets as well.
- `-s, --sort_estimands`: Sort estimands to minimize cache usage (A brute force approach will be used, resulting in exponentially long sorting time).
"""
function tmle(dataset::String;
    estimands::String="factorialATE", 
    estimators::String="glmnet",
    verbosity::Int=0, 
    outputs::Outputs=Outputs(),
    chunksize::Int=100,
    rng_seed::Int=123,
    cache_strategy::String="release-unusable",
    sort_estimands::Bool=false,
    save_sample_ids=false,
    pvalue_threshold=nothing,
    prevalence_file=nothing,
    prevalence_mode="sampling",
    output_downsampled_datasets::Bool=true
    )
    runner = Runner(dataset;
        estimands_config=estimands, 
        estimators_spec=estimators, 
        verbosity=verbosity, 
        outputs=outputs, 
        chunksize=chunksize,
        rng_seed=rng_seed,
        cache_strategy=cache_strategy,
        sort_estimands=sort_estimands,
        save_sample_ids=save_sample_ids,
        pvalue_threshold=pvalue_threshold,
        prevalence_file=prevalence_file,
        prevalence_mode=prevalence_mode,
	output_downsampled_datasets=output_downsampled_datasets
    )
    runner()
    verbosity >= 1 && @info "Done."
    return
end
