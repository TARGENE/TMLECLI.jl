module TestSAL

using TargetedEstimation
using Test
using MLJ
using StableRNGs
using EvoTrees


@testset "Test ConstantBasis" begin
    n = 100
    X, y = make_regression(n)
    model = TargetedEstimation.ConstantBasis()
    mach = machine(model, X, y)
    fit!(mach, verbosity=0)
    @test predict(mach, X) == fill(0., n)
    model.value = 1.
    fit!(mach, verbosity=0)
    @test predict(mach, X) == fill(1., n)
    # Automatic conversion of Int to Float
    model.value = 2
    @test model.value isa Float64
end

@testset "Test misc" begin
    model = SALRegressor()
    y = [1., 2., 3.]
    constant_basis = TargetedEstimation.constant_basis(model, y)
    @test constant_basis.value == mean(y)
    
end

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