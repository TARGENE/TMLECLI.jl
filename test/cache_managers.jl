module TestRunner

using TargetedEstimation
using Test
using TMLE

@testset "Test NoCacheManager" begin
    cache_manager = TargetedEstimation.NoCacheManager()
    cache_manager.cache["Toto"] = 1
    cache_manager.cache["Tata"] = 2
    TargetedEstimation.release!(cache_manager, nothing)
    @test cache_manager.cache == Dict()
    # Check this does not throw
    TargetedEstimation.release!(cache_manager, nothing)
end

@testset "Test MaxSizeCacheManager" begin
    cache_manager = TargetedEstimation.MaxSizeCacheManager(3)
    Y_T₁T₂ = TMLE.ConditionalDistribution(:Y, (:T₁, :T₂, :W))
    cache_manager.cache[Y_T₁T₂] = 1
    T₁_W = TMLE.ConditionalDistribution(:T₁, (:W,))
    cache_manager.cache[T₁_W] = 1
    T₂_W = TMLE.ConditionalDistribution(:T₂, (:W,))
    cache_manager.cache[T₂_W] = 1
    η = TMLE.CMRelevantFactors(
        Y_T₁T₂,
        (T₁_W, T₂_W)
    )
    cache_manager.cache[η] = 1
    cache_manager.cache[:last_fluctuation] = 1
    length(cache_manager.cache) == 5
    TargetedEstimation.release!(cache_manager, nothing)
    # CMRelevantFactors and fluctuation dropped
    @test cache_manager.cache == Dict(
        TMLE.ConditionalDistribution(:Y, (:T₁, :T₂, :W)) => 1,
        TMLE.ConditionalDistribution(:T₂, (:W,))         => 1,
        TMLE.ConditionalDistribution(:T₁, (:W,))         => 1
    )
end

@testset "Test ReleaseUnusableCacheManager" begin
    estimands = [
        ATE(
            outcome=:Y, 
            treatment_values=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
            treatment_confounders=(T₁=[:W], T₂=[:W])
        ),
        ATE(
            outcome=:Y, 
            treatment_values=(T₁=(case=1, control=0), T₂=(case=2, control=0)),
            treatment_confounders=(T₁=[:W], T₂=[:W])
        ),
        ATE(
            outcome=:Y, 
            treatment_values=(T₁=(case=1, control=0),),
            treatment_confounders=(T₁=[:W],)
        ),
        ATE(
            outcome=:Ynew, 
            treatment_values=(T₃=(case=1, control=0),),
            treatment_confounders=(T₃=[:W],)
        )
    ]
    η_counts = TMLE.nuisance_function_counts(estimands)
    cache_manager = TargetedEstimation.ReleaseUnusableCacheManager(η_counts)
    # Estimation of the first estimand will fill the cache with the following
    Y_T₁T₂ = TMLE.ConditionalDistribution(:Y, (:T₁, :T₂, :W))
    cache_manager.cache[Y_T₁T₂] = 1
    T₁_W = TMLE.ConditionalDistribution(:T₁, (:W,))
    cache_manager.cache[T₁_W] = 1
    T₂_W = TMLE.ConditionalDistribution(:T₂, (:W,))
    cache_manager.cache[T₂_W] = 1
    η = TMLE.CMRelevantFactors(
        Y_T₁T₂,
        (T₁_W, T₂_W)
    )
    cache_manager.cache[η] = 1
    cache_manager.cache[:last_fluctuation] = 1
    @test length(cache_manager.cache) == 5
    # After estimation of the first estimand, the fluctuation and composite factor are released
    TargetedEstimation.release!(cache_manager, estimands[1])
    @test cache_manager.cache == Dict(
        TMLE.ConditionalDistribution(:Y, (:T₁, :T₂, :W)) => 1,
        TMLE.ConditionalDistribution(:T₂, (:W,))         => 1,
        TMLE.ConditionalDistribution(:T₁, (:W,))         => 1
    )

    # Estimation of the second estimand will restore the composite factor
    cache_manager.cache[η] = 1
    cache_manager.cache[:last_fluctuation] = 1
    # Y_T₁T₂ and T₂_W are no longer needed
    TargetedEstimation.release!(cache_manager, estimands[2])
    @test cache_manager.cache == Dict(TMLE.ConditionalDistribution(:T₁, (:W,)) => 1)

    # Estimation of the third estimand will fill the cache with the following
    Y_T₁ = TMLE.ConditionalDistribution(:Y, (:T₁, :W))
    cache_manager.cache[Y_T₁] = 1
    η = TMLE.CMRelevantFactors(
        Y_T₁,
        (T₁_W, )
    )
    cache_manager.cache[η] = 1
    cache_manager.cache[:last_fluctuation] = 1
    # Y_T₁ and T₁_W are no longer needed
    TargetedEstimation.release!(cache_manager, estimands[3])
    @test cache_manager.cache == Dict()
    # Check this does not throw
    TargetedEstimation.release!(cache_manager, estimands[1])
end

end

true