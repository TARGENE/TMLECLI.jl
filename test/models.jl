module TestModels

using Test
using TargetedEstimation
using MLJ
using DataFrames
using RCall
using StableRNGs

@testset "Test InteractionTransformer" begin
    X = (rs1234=[1, 2, 3], rs455=[4, 5, 6], rs4489=[7, 8, 9], rstoto=[1, 2, 3])
    t = TargetedEstimation.InteractionTransformer(r"^rs[0-9]+")
    mach = machine(t, X)
    fit!(mach, verbosity=0)
    Xt = MLJ.transform(mach, X)

    @test Xt == (
        rs1234 = [1, 2, 3],
        rs455 = [4, 5, 6],
        rs4489 = [7, 8, 9],
        rstoto = [1, 2, 3],
        rs1234_rs455 = [4.0, 10.0, 18.0],
        rs1234_rs4489 = [7.0, 16.0, 27.0],
        rs455_rs4489 = [28.0, 40.0, 54.0]
    )
    @test mach.fitresult.ninter == 3
    @test mach.fitresult.interaction_pairs == [:rs1234 => :rs455, :rs1234 => :rs4489, :rs455 => :rs4489]

end

@testset "Test InteractionLM" begin
    n = 100
    X = (rs1234=rand(n), rs455=rand(n), rstoto=rand(n))
    # Classifier
    y = categorical(rand([0,1], n))
    mach = machine(
        TargetedEstimation.InteractionLMClassifier(lambda=10.),
        X,
        y
    )
    fit!(mach, verbosity=0)
    fp = fitted_params(mach)
    @test fp.interaction_transformer.fitresult.interaction_pairs == [:rs1234 => :rs455]
    @test size(predict(mach), 1) == n 
    # Regressor 
    y = rand(n)
    mach = machine(
        TargetedEstimation.InteractionLMRegressor(lambda=10),
        X,
        y
    )
    fit!(mach, verbosity=0)
    fp = fitted_params(mach)
    @test fp.interaction_transformer.fitresult.interaction_pairs == [:rs1234 => :rs455]
    @test predict(mach) isa Vector{Float64}

    @test target_scitype(TargetedEstimation.InteractionLMRegressor()) == AbstractVector{<:MLJ.Continuous}
    @test target_scitype(TargetedEstimation.InteractionLMClassifier()) == AbstractVector{<:MLJ.Finite}

end

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