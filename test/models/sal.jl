module TestSAL

using TargetedEstimation
using Test
using MLJ
using StableRNGs
using EvoTrees


@testset "Test SALRegressor" begin
    rng = StableRNG(123)
    X, y = make_regression(100, 3, rng=rng)
    sal = SALRegressor()
    mach = machine(sal, X, y)
    fit!(mach, verbosity=0)
    @test size(mach.fitresult.gbt_machs, 1) == sal.n_bases
    @test mach.fitresult.lasso_mach.state == 1

    # Check training loss decreases by increasing 
    # the number of bases
    K = 5
    rmses = Vector{Float64}(undef, K)
    for k in 1:K
        sal = SALRegressor(n_bases=k)
        mach = machine(sal, X, y)
        fit!(mach, verbosity=0)
        rmses[k] = TargetedEstimation.loss(sal, predict(mach), y)
    end
    @test rmses[1] > rmses[2] > rmses[3] > rmses[4] > rmses[5]

    # Test residuals
    @test TargetedEstimation.residuals(sal, [1, 2, 3], [1, 2, 3]) == [0, 0, 0]

end


end

true