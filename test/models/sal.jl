module TestSAL

using TargetedEstimation
using Test
using MLJ
using StableRNGs
using EvoTrees

@testset "Test SALRegressor" begin
    rng = StableRNG(123)
    X, y = make_regression(1000, 3; rng=rng)

    sal = TargetedEstimation.SALRegressor()
    mach = machine(sal, X, y)
    fit!(mach, verbosity=0)
    MLJBase.predict(mach)
    sal_pe = evaluate!(mach, measure=rmse)

    mach = machine(EvoTreeRegressor(), X, y)
    evotree_pe = evaluate!(mach, measure=rmse)

    iterated_sal = IteratedModel(
        model=sal,
        resampling=Holdout(),
        measures=rmse,
        controls=[Step(1),
                Patience(2),
                    NumberLimit(100)],
        iteration_parameter=:n_iter,
        retrain=true)

    mach = machine(iterated_sal, X, y)
    fit!(mach, verbosity=1)

    iterated_model = IteratedModel(model=EvoTreeRegressor(),
        resampling=Holdout(),
        measures=rmse,
        controls=[Step(2),
                  Patience(2),
                  NumberLimit(100)],
        retrain=true)
    mach = machine(iterated_model, X, y)
    fit!(mach, verbosity=1)
end

end

true