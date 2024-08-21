module TestRestrictedInteractionTransformer

using Test
using TmleCLI
using Random
using MLJBase
using Tables

@testset "Test RestrictedInteractionTransformer" begin
    n = 10
    X = (
        rs1234 = rand(n),
        rs4567 = rand(n),
        sex = rand(n),
        Age = rand(n),
        PC_1 = rand(n),
        PC_2 = rand(n),
    )

    # Test with primary_variables only
    model = RestrictedInteractionTransformer(;
        order=3, 
        primary_variables=[:rs1234, :sex],
    )
    @test input_scitype(model) == Table(Continuous)
    @test output_scitype(model) == Table(Continuous)
    mach = machine(model, X)
    fit!(mach, verbosity=0)
    interactions = fitted_params(mach).fitresult
    expected_interactions = [
        [:rs1234, :sex],
        [:rs1234, :rs4567],
        [:rs1234, :Age],
        [:rs1234, :PC_1],
        [:rs1234, :PC_2],
        [:sex, :rs4567],
        [:sex, :Age],
        [:sex, :PC_1],
        [:sex, :PC_2],
        [:rs1234, :sex, :rs4567],
        [:rs1234, :sex, :Age],
        [:rs1234, :sex, :PC_1],
        [:rs1234, :sex, :PC_2]
    ]
    @test expected_interactions == interactions
    Xt = MLJBase.transform(mach, X)
    feature_names = TmleCLI.feature_names(interactions)
    @test Tables.columnnames(Xt) == tuple(Tables.columnnames(X)..., feature_names...)
    @test Xt[Symbol("rs1234_&_sex_&_PC_2")] == X[:rs1234].*X[:sex].*X[:PC_2]

    # Test with primary_variables_patterns only
    model = RestrictedInteractionTransformer(;
        order=2, 
        primary_variables_patterns=[r"^rs[0-9]+", r"sex"],
    )
    mach = machine(model, X)
    fit!(mach, verbosity=0)
    interactions = fitted_params(mach).fitresult
    expected_interactions = [
        [:rs1234, :rs4567],
        [:rs1234, :sex],
        [:rs4567, :sex],
        [:rs1234, :Age],
        [:rs1234, :PC_1],
        [:rs1234, :PC_2],
        [:rs4567, :Age],
        [:rs4567, :PC_1],
        [:rs4567, :PC_2],
        [:sex, :Age],
        [:sex, :PC_1],
        [:sex, :PC_2],
    ]
    @test expected_interactions == interactions
    Xt = MLJBase.transform(mach, X)
    feature_names = TmleCLI.feature_names(interactions)
    @test Tables.columnnames(Xt) == tuple(Tables.columnnames(X)..., feature_names...)

    # Test with both primary_variables and primary_variables_patterns
    model = RestrictedInteractionTransformer(;
        order=3,
        primary_variables=[:sex],
        primary_variables_patterns=[r"^rs[0-9]+"],
    )
    mach = machine(model, X)
    fit!(mach, verbosity=0)
    interactions = fitted_params(mach).fitresult
    expected_interactions = [
        [:sex, :rs1234],
        [:sex, :rs4567],
        [:rs1234, :rs4567],
        [:sex, :Age],
        [:sex, :PC_1],
        [:sex, :PC_2],
        [:rs1234, :Age],
        [:rs1234, :PC_1],
        [:rs1234, :PC_2],
        [:rs4567, :Age],
        [:rs4567, :PC_1],
        [:rs4567, :PC_2],
        [:sex, :rs1234, :rs4567],
        [:sex, :rs1234, :Age],
        [:sex, :rs1234, :PC_1],
        [:sex, :rs1234, :PC_2],
        [:sex, :rs4567, :Age],
        [:sex, :rs4567, :PC_1],
        [:sex, :rs4567, :PC_2],
        [:rs1234, :rs4567, :Age],
        [:rs1234, :rs4567, :PC_1],
        [:rs1234, :rs4567, :PC_2],
    ]
    @test expected_interactions == interactions
    Xt = MLJBase.transform(mach, X)
    feature_names = TmleCLI.feature_names(interactions)
    @test Tables.columnnames(Xt) == tuple(Tables.columnnames(X)..., feature_names...)

    # Invalid column
    X = (A=["A1", "A2", "A1"], B=[1, 2, 3])
    model = RestrictedInteractionTransformer(;
        order=2,
        primary_variables=[:A, :B],
    )
    mach = machine(model, X, scitype_check_level=0)
    @test_throws TmleCLI.InvalidColumnError("A") fit!(mach, verbosity=0)
end

end

true