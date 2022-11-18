using TargetedEstimation
using Test
using MLJ
using StableRNGs


@testset "Test SAL" begin
    rng = StableRNG(123)
    sal = TargetedEstimation.SALRegressor()
    X, y = make_regression(100, 2; rng=rng)
    mach = machine(sal, X, y)
    fit!(mach, verbosity=1)
end