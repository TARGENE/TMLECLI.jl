module TestGridSearchEvoTree

using MLJ
using DataFrames
using EvoTrees
using StableRNGs

@testset "Test parsing" begin
    

end

@testset "Test Regressor" begin
    n = 1000
    X, y = make_regression(n)

    model = GridSearchEvoTreeRegressor(;nrounds=100, rng=StableRNG(123), max_depth="3, 7", lambda="1e-5,10,log")
    mach = machine(model, X, y)
    fit!(mach)
    evaluate!(mach, verbosity=1, measure=rmse, resampling=CV(rng=StableRNG(123)))

    mach_bis = machine(EvoTreeRegressor(nrounds=100, rng=StableRNG(123)), X, y)
    fit!(mach_bis)
    evaluate!(mach_bis, verbosity=1, measure=rmse, resampling=CV(rng=StableRNG(123)))

end


end