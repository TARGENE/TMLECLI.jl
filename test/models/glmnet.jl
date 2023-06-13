module TestGLMNet

using Test
using TargetedEstimation
using MLJ
using StableRNGs

@testset "Test misc" begin
    n = 10
    rng = StableRNG(123)
    X = rand(rng, n, 3)
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    folds = TargetedEstimation.getfolds(CV(), X, y)
    @test folds == [1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    folds = TargetedEstimation.getfolds(StratifiedCV(nfolds=3), X, y)
    @test folds == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
end

@testset "Test GLMNetModel" begin
    ## The following will test both the fit/predict APIs
    # Regressor
    rng = StableRNG(123)
    n, p = 1000, 5
    X, y = make_regression(n, p, rng=rng)
    net = TargetedEstimation.GLMNetRegressor(resampling=CV(nfolds=3), rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=rmse, resampling=CV(rng=rng), verbosity=0)
    @test pe.measurement[1] < 0.1
    
    # Binary outcome
    rng = StableRNG(123)
    X, y = make_moons(n, rng=rng)
    net = TargetedEstimation.GLMNetClassifier(rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=log_loss, resampling=StratifiedCV(rng=rng), verbosity=0)
    @test pe.measurement[1] < 0.180

    # Multivariate outcome
    rng = StableRNG(123)
    X, y = make_blobs(n, rng=rng)
    net = TargetedEstimation.GLMNetClassifier(resampling=StratifiedCV(nfolds=3), rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=[log_loss], resampling=StratifiedCV(rng=rng), verbosity=0)
    @test pe.measurement[1] < 0.008
end

@testset "Test InteractionGLMNet" begin
    # Regression
    rng = StableRNG(123)
    n, p = 1000, 5
    X, y = make_regression(n, p, rng=rng)
    net = TargetedEstimation.InteractionGLMNetRegressor(order=3, nfolds=3)
    mach = machine(net, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach) isa Vector{Float64}
    # Classification
    rng = StableRNG(123)
    X, y = make_blobs(n, rng=rng)
    net = TargetedEstimation.InteractionGLMNetClassifier(order=2, rng=rng, cache=true)
    mach = machine(net, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach) isa MLJ.UnivariateFiniteVector
end

@testset "Test RestrictedInteractionGLMNet" begin
    # Regression
    rng = StableRNG(123)
    n, p = 1000, 5
    X, y = make_regression(n, p, rng=rng)
    net = TargetedEstimation.RestrictedInteractionGLMNetRegressor(
        order=3, 
        primary_patterns=["x1"],
        nfolds=3
    )
    mach = machine(net, X, y)
    fit!(mach, verbosity=0)
    fp = fitted_params(mach).interaction_transformer.fitresult
    @test fp == [
        [:x1, :x2],
        [:x1, :x3],
        [:x1, :x4],
        [:x1, :x5]
    ]
    @test predict(mach) isa Vector{Float64}
    # Classification
    rng = StableRNG(123)
    X, y = make_blobs(n, rng=rng)
    net = TargetedEstimation.RestrictedInteractionGLMNetClassifier(order=2, rng=rng, cache=true)
    mach = machine(net, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach) isa MLJ.UnivariateFiniteVector
    fp = fitted_params(mach).interaction_transformer.fitresult
    @test fp == []
end

end

true