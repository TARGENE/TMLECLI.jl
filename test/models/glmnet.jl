module TestGLMNet

using Test
using TargetedEstimation
using MLJ
using StableRNGs

@testset "Test GLMNetModel" begin
    # Regressor
    rng = StableRNG(123)
    n, p = 1000, 5
    X, y = make_regression(n, p, rng=rng)
    net = TargetedEstimation.GLMNetRegressor(rng=rng)
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

end

true