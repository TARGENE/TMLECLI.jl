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
end

@testset "Test MaxSizeCacheManager" begin
    cache_manager = TargetedEstimation.MaxSizeCacheManager(3)
    cache_manager.cache["Toto"] = 1
    cache_manager.cache["Tata"] = 2
    TargetedEstimation.release!(cache_manager, nothing)
    @test cache_manager.cache == Dict("Toto" => 1, "Tata" => 2)
    cache_manager.cache["Titi"] = 3
    cache_manager.cache["Tutu"] = 4
    @test length(cache_manager.cache) == 4
    TargetedEstimation.release!(cache_manager, nothing)
    @test length(cache_manager.cache) == 3
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
    η_counts = TMLE.nuisance_counts(estimands)
    cache_manager = TargetedEstimation.ReleaseUnusableCacheManager(η_counts)
    # Estimation of the first estimand will fill the cache with the following
    Y_T₁T₂ = TMLE.ConditionalDistribution(:Y, (:T₁, :T₂, :W))
    cache_manager.cache[Y_T₁T₂] = 1
    T₁_W = TMLE.ConditionalDistribution(:T₁, (:W,))
    cache_manager.cache[T₁_W] = 1
    T₂_W = TMLE.ConditionalDistribution(:T₂, (:W,))
    cache_manager.cache[T₂_W] = 1
    cache_manager.cache[:last_fluctuation] = 1
    @test length(cache_manager.cache) == 4
    # After estimation of the first estimand, only the fluctuation is released
    TargetedEstimation.release!(cache_manager, estimands[1])
    @test length(cache_manager.cache) == 3

    # Estimation of the second estimand will not result in further nuisance functions
    # Y_T₁T₂ and T₂_W are no longer needed
    TargetedEstimation.release!(cache_manager, estimands[2])
    @test length(cache_manager.cache) == 1
    @test !haskey(cache_manager.cache, T₂_W)
    @test !haskey(cache_manager.cache, Y_T₁T₂)
    @test haskey(cache_manager.cache, T₁_W)

    # Estimation of the third estimand will fill the cache with the following
    Y_T₁ = TMLE.ConditionalDistribution(:Y, (:T₁, :W))
    cache_manager.cache[Y_T₁] = 1
    # Y_T₁ and T₁_W are no longer needed
    TargetedEstimation.release!(cache_manager, estimands[3])
    @test cache_manager.cache == Dict()


end

end

true