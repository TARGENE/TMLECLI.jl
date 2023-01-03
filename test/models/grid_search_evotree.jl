module TestGridSearchEvoTree

using MLJ
using TargetedEstimation
using EvoTrees
using Test

@testset "Test parsing" begin
    model = EvoTreeClassifier(nrounds=30, eta=0.1)
    ranges = []
    TargetedEstimation.update_model_or_ranges!(model, ranges, "nrounds", 10)
    @test model.nrounds == 10

    TargetedEstimation.update_model_or_ranges!(model, ranges, "max_depth", "3, 9, linear")
    @test ranges[1].lower == 3
    @test ranges[1].upper == 9
    @test ranges[1].scale == :linear

    TargetedEstimation.update_model_or_ranges!(model, ranges, "eta", 0.3)
    @test model.eta == 0.3f0

    TargetedEstimation.update_model_or_ranges!(model, ranges, "lambda", "1e-5, 100, log")
    @test ranges[2].lower == 1e-5
    @test ranges[2].upper == 100
    @test ranges[2].scale == :log
end

@testset "Test GridSearchEvoTreeRegressor" begin
    n = 1000
    X, y = make_regression(n)

    model = GridSearchEvoTreeRegressor(;
        nrounds=100, 
        resampling=Dict(:type=>"Holdout"), 
        goal=25,
        max_depth="3, 7", 
        lambda="1e-5,10,log"
    )
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    @test size(report(mach).history, 1) == 25
    @test fitted_params(mach).best_model isa EvoTreeRegressor
end

@testset "Test GridSearchEvoTreeClassifier" begin
    n = 1000
    X, y = make_blobs(n)

    model = GridSearchEvoTreeClassifier(;
        nrounds=100, 
        resampling=Dict(:type=>"Holdout"), 
        goal=5,
        max_depth="3, 7", 
        lambda="1e-5,10,log",
        cache=true
    )
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    @test size(report(mach).history, 1) == 4
    @test fitted_params(mach).best_model isa EvoTreeClassifier
end


end