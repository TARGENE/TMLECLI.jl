abstract type CacheManager end

struct ReleaseUnusableCacheManager <: CacheManager
    cache::Dict
    η_counts::Dict
    ReleaseUnusableCacheManager(η_counts) = new(Dict(), η_counts)
end

function release!(cache_manager::ReleaseUnusableCacheManager, Ψ)
    # Always drop fluctuations
    haskey(cache_manager.cache, :last_fluctuation) && pop!(cache_manager.cache, :last_fluctuation)

    # Drop Basic nuisance functions
    for η in TMLE.nuisance_functions_iterator(Ψ)
        cache_manager.η_counts[η] -= 1
        if cache_manager.η_counts[η] == 0
            delete!(cache_manager.cache, η)
        end
    end

    # Drop aggregate nuisance function
    for η in keys(cache_manager.cache)
        if η isa TMLE.CMRelevantFactors
            delete!(cache_manager.cache, η)
        end
    end
end

struct MaxSizeCacheManager <: CacheManager
    cache::Dict
    max_size::Int
    MaxSizeCacheManager(max_size) = new(Dict(), max_size)
end

function release!(cache_manager::MaxSizeCacheManager, Ψ)
    # Prioritize the release of the last fluctuation
    if haskey(cache_manager.cache, :last_fluctuation)
        pop!(cache_manager.cache, :last_fluctuation)
    end
    # Drop aggregate nuisance function
    for η in keys(cache_manager.cache)
        if η isa TMLE.CMRelevantFactors
            delete!(cache_manager.cache, η)
        end
    end
    # Drop the rest randomly until the size is acceptable
    while length(cache_manager.cache) > cache_manager.max_size
        pop!(cache_manager.cache)
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
        return ReleaseUnusableCacheManager(TMLE.nuisance_function_counts(estimands))
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



