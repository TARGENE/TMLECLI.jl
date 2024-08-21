module TestGLMNet

using Test
using TmleCLI
using MLJ
using StableRNGs

@testset "Test misc" begin
    n = 10
    rng = StableRNG(123)
    X = rand(rng, n, 3)
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    folds = TmleCLI.getfolds(CV(), X, y)
    @test folds == [1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    folds = TmleCLI.getfolds(StratifiedCV(nfolds=3), X, y)
    @test folds == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
end

@testset "Test GLMNetModel" begin
    ## The following will test both the fit/predict APIs
    # Regressor
    rng = StableRNG(123)
    n, p = 1000, 5
    X, y = make_regression(n, p, rng=rng)
    net = TmleCLI.GLMNetRegressor(resampling=CV(nfolds=3), rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=rmse, resampling=CV(rng=rng), verbosity=0)
    @test pe.measurement[1] < 0.1
    
    # Binary outcome
    rng = StableRNG(123)
    X, y = make_moons(n, rng=rng)
    net = TmleCLI.GLMNetClassifier(rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=log_loss, resampling=JointStratifiedCV(resampling=StratifiedCV(rng=rng)), verbosity=0)
    @test pe.measurement[1] < 0.180

    # Multivariate outcome
    rng = StableRNG(123)
    X, y = make_blobs(n, rng=rng)
    net = TmleCLI.GLMNetClassifier(resampling=StratifiedCV(nfolds=3), rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=[log_loss], resampling=StratifiedCV(rng=rng), verbosity=0)
    @test pe.measurement[1] < 0.008
end

end

true