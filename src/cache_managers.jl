abstract type CacheManager end

struct ReleaseUnusableCacheManager <: CacheManager
    cache::Dict
    η_counts::Dict
    ReleaseUnusableCacheManager(η_counts) = new(Dict(), η_counts)
end

function release!(cache_manager::ReleaseUnusableCacheManager, Ψ)
    # Always drop fluctuations
    haskey(cache_manager.cache, :last_fluctuation) && pop!(cache_manager.cache, :last_fluctuation)

    η = TMLE.get_relevant_factors(Ψ)
    # Propensity scores
    for ps in η.propensity_score
        cache_manager.η_counts[ps] -= 1
        if cache_manager.η_counts[ps] == 0
            pop!(cache_manager.cache, ps)
        end
    end
    # Outcome Mean
    cache_manager.η_counts[η.outcome_mean] -= 1
    if cache_manager.η_counts[η.outcome_mean] == 0
        pop!(cache_manager.cache, η.outcome_mean)
    end
end

struct MaxSizeCacheManager <: CacheManager
    cache::Dict
    max_size::Int
    MaxSizeCacheManager(max_size) = new(Dict(), max_size)
end

function release!(cache_manager::MaxSizeCacheManager, Ψ)
    while length(cache_manager.cache) > cache_manager.max_size
        # Prioritize the release of the last fluctuation
        if haskey(cache_manager.cache, :last_fluctuation)
            pop!(cache_manager.cache, :last_fluctuation)
        else
            pop!(cache_manager.cache)
        end
    end
end

struct NoCacheManager <: CacheManager
    cache::Dict
    NoCacheManager() = new(Dict())
end

function release!(cache_manager::NoCacheManager, Ψ)
    empty!(cache_manager.cache)
end

function make_cache_manager(estimands, string)
    if string == "release-unusable"
        return ReleaseUnusableCacheManager(TMLE.nuisance_counts(estimands))
    elseif string == "no-cache"
        return NoCacheManager()
    else
        maxsize = try parse(Int, string) 
            catch E
                throw(ArgumentError(string("Could not convert the provided cache value to an integer: ", string)))
            end
        return MaxSizeCacheManager(maxsize)
    end
end



