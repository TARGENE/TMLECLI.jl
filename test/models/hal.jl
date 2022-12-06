module TestHAL

using Test
using TargetedEstimation
using StableRNGs
using DataFrames
using MLJ
using RCall

@testset "Test SNPInteractionHAL" begin
    n = 100
    rng = StableRNG(123)
    X = DataFrame(rs1234=rand(rng, n), rs455=rand(rng, n), PC1=rand(rng, n), PC_2=rand(rng, n))
    # formula hal
    formula = TargetedEstimation.build_formula_hal(X, Regex("^rs[0-9]+"))
    @test formula == "~h(.) + h(rs1234, .) + h(rs455, .)"
    # Classifier
    y = categorical(rand([0,1], n))
    pairwise_snp_hal = TargetedEstimation.SNPInteractionHALClassifier(
        lambda=1, 
        smoothness_orders=1, 
        num_knots=[10, 5], 
        cv_select=false
    )
    mach = machine(
        pairwise_snp_hal,
        X,
        y
    )
    fit!(mach, verbosity=0)
    # Get the number of basis for comparison with default degree 2
    fr = fitted_params(mach).fitresult[1]
    @rput fr
    R"nb_basis = length(fr$basis_list)"
    @rget nb_basis
    @test nb_basis == 96

    @test predict(mach, X) isa MLJ.UnivariateFiniteVector
    # default degree 2
    base_hal = TargetedEstimation.HALClassifier(
        lambda=1, 
        max_degree=2, 
        smoothness_orders=1, 
        num_knots=[10, 5], 
        cv_select=false
    )
    mach = machine(
        base_hal,
        X,
        y
    )
    fit!(mach, verbosity=0)
    fr = fitted_params(mach).fitresult[1]
    @rput fr
    R"nb_basis = length(fr$basis_list)"
    @rget nb_basis
    @test nb_basis == 157

    # Regressor 
    y = rand(rng, n)
    pairwise_snp_hal = TargetedEstimation.SNPInteractionHALRegressor(
        lambda=1, 
        smoothness_orders=1, 
        num_knots=[10, 5], 
        cv_select=false
    )
    mach = machine(
        pairwise_snp_hal,
        X,
        y
    )
    fit!(mach, verbosity=0)
    @test predict(mach, X) isa Vector
end

end

true