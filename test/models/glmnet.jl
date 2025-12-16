module TestGLMNet

using Test
using TMLECLI
using MLJ
using StableRNGs

@testset "Test misc" begin
    n = 10
    rng = StableRNG(123)
    X = rand(rng, n, 3)
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    folds = TMLECLI.getfolds(CV(), X, y)
    @test folds == [1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    folds = TMLECLI.getfolds(StratifiedCV(nfolds=3), X, y)
    @test folds == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
end

@testset "Test GLMNetModel" begin
    ## The following will test both the fit/predict APIs
    # Regressor
    rng = StableRNG(123)
    n, p = 1000, 5
    X, y = make_regression(n, p, rng=rng)
    net = TMLECLI.GLMNetRegressor(resampling=CV(nfolds=3), rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=rmse, resampling=CV(rng=rng), verbosity=0)
    @test pe.measurement[1] < 0.1
    
    # Binary outcome
    rng = StableRNG(123)
    X, y = make_moons(n, rng=rng)
    net = TMLECLI.GLMNetClassifier(rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=log_loss, resampling=JointStratifiedCV(resampling=StratifiedCV(rng=rng)), verbosity=0)
    @test pe.measurement[1] < 0.180

    # Multivariate outcome
    rng = StableRNG(123)
    X, y = make_blobs(n, rng=rng)
    net = TMLECLI.GLMNetClassifier(resampling=StratifiedCV(nfolds=3), rng=rng)
    mach = machine(net, X, y)
    pe = evaluate!(mach, measure=[log_loss], resampling=StratifiedCV(rng=rng), verbosity=0)
    @test pe.measurement[1] < 0.008
end


@testset "Test case-control weights" begin
    rng = StableRNG(123)
    X, y = make_moons(1000, rng=rng)
    weights = ifelse.(y .== 1, 0.1, 2.0)
    @test TMLECLI.supports_weights(TMLECLI.GLMNetClassifier()) == true
    
    net1 = TMLECLI.GLMNetClassifier(rng=rng)
    net2 = TMLECLI.GLMNetClassifier(rng=rng)
    
    weighted_mach = machine(net1, X, y, weights) 
    unweighted_mach = machine(net2, X, y)
    fit!(weighted_mach)
    fit!(unweighted_mach)
    
    # Compare results
    weighted_glmnet = fitted_params(weighted_mach).fitresult.glmnetcv
    unweighted_glmnet = fitted_params(unweighted_mach).fitresult.glmnetcv
    
    weighted_preds = predict(weighted_mach, X)
    unweighted_preds = predict(unweighted_mach, X)
    weighted_probs = pdf.(weighted_preds, 1)
    unweighted_probs = pdf.(unweighted_preds, 1)
    
    @test !isapprox(weighted_probs, unweighted_probs, rtol=1e-8)
end

end

true