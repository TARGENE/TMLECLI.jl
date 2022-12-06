module TestGLMNet

using Test
using TargetedEstimation
using MLJ
using StableRNGs

@testset "Test GLMNetModel" begin
    ## The following will test both the fit/predict APIs
    # Regressor
    rng = StableRNG(123)
    n, p = 1000, 5
    X, y = make_regression(n, p, rng=rng)
    net = TargetedEstimation.GLMNetRegressor(nfolds=3, rng=rng)
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
    net = TargetedEstimation.GLMNetClassifier(rng=rng)
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
    net = TargetedEstimation.InteractionGLMNetClassifier(order=2, rng=rng)
    mach = machine(net, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach) isa MLJ.UnivariateFiniteVector

end

end

true