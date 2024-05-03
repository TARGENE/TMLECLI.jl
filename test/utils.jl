module TestUtils

using Test
using TargetedEstimation
using TMLE
using DataFrames
using CSV
using MLJLinearModels
using CategoricalArrays
using MLJBase

check_type(treatment_value, ::Type{T}) where T = @test treatment_value isa T

check_type(treatment_values::NamedTuple, ::Type{T}) where T = 
    @test treatment_values.case isa T && treatment_values.control isa T 

TESTDIR = joinpath(pkgdir(TargetedEstimation), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test convert_treatment_values" begin
    treatment_types = Dict(:T₁=> Union{Missing, Bool}, :T₂=> Int)
    newT = TargetedEstimation.convert_treatment_values((T₁=1,), treatment_types)
    @test newT isa Vector{Bool}
    @test newT == [1]

    newT = TargetedEstimation.convert_treatment_values((T₁=(case=1, control=0.),), treatment_types)
    @test newT isa Vector{NamedTuple{(:case, :control), Tuple{Bool, Bool}}}
    @test newT == [(case = true, control = false)]

    newT = TargetedEstimation.convert_treatment_values((T₁=(case=1, control=0.), T₂=(case=true, control=0)), treatment_types)
    @test newT isa Vector{NamedTuple{(:case, :control)}}
    @test newT == [(case = true, control = false), (case = 1, control = 0)]
end

@testset "Test instantiate_config" for extension in ("yaml", "json")
    # Write estimands file
    filename = "statistical_estimands.$extension"
    eval(Meta.parse("TMLE.write_$extension"))(filename, statistical_estimands_only_config())

    dataset = DataFrame(T1 = [1., 0.], T2=[true, false])
    config = TargetedEstimation.instantiate_config(filename)
    estimands = TargetedEstimation.proofread_estimands(config, dataset)
    for estimand in estimands
        if haskey(estimand.treatment_values, :T1)
            check_type(estimand.treatment_values.T1, Float64)
        end
        if haskey(estimand.treatment_values, :T2)
            check_type(estimand.treatment_values.T2, Bool)
        end
    end
    # Clean estimands file
    rm(filename)
end

@testset "Test factorialATE" begin
    dataset = DataFrame(C=[1, 2, 3, 4],)
    @test_throws ArgumentError TargetedEstimation.instantiate_estimands("factorialATE", dataset)
    dataset.T = [0, 1, missing, 2]
    @test_throws ArgumentError TargetedEstimation.instantiate_estimands("factorialATE", dataset)
    dataset.Y = [0, 1, 2, 2]
    dataset.W1 = [1, 1, 1, 1]
    dataset.W_2 = [1, 1, 1, 1]
    composedATE = TargetedEstimation.instantiate_estimands("factorialATE", dataset)[1]
    @test composedATE.args == (
        TMLE.StatisticalATE(:Y, (T = (case = 1, control = 0),), (T = (:W1, :W_2),), ()),
        TMLE.StatisticalATE(:Y, (T = (case = 2, control = 1),), (T = (:W1, :W_2),), ())
    )
end
@testset "Test coerce_types!" begin
    dataset = DataFrame(
        Ycont  = [1.1, 2.2, missing, 3.5, 6.6, 0., 4.],
        Ybin = [1., 0., missing, 1., 0, 0, 0],
        Ycount = [1, 0., missing, 1, 2, 0, 3],
        T₁ = [1, 0, missing, 0, 0, 0, missing],
        T₂ = [missing, "AC", "CC", "CC", missing, "AA", "AA"],
        W₁ = [1., 0., 0., 1., 0., 1, 1],
        W₂ = [missing, 0., 0., 0., 0., 0., 0.],
        C = [1, 2, 3, 4, 5, 6, 6]
    )
    # Continuous Outcome
    Ψ = IATE(
        outcome=:Ycont,
        treatment_values=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        treatment_confounders=(T₁=[:W₁, :W₂], T₂=[:W₁, :W₂]),
    )
    TargetedEstimation.coerce_types!(dataset, Ψ)
    @test scitype(dataset.T₁) == AbstractVector{Union{Missing, OrderedFactor{2}}}
    @test scitype(dataset.T₂) == AbstractVector{Union{Missing, Multiclass{3}}}
    @test scitype(dataset.Ycont) == AbstractVector{Union{Missing, MLJBase.Continuous}}
    @test scitype(dataset.W₁) == AbstractVector{OrderedFactor{2}}
    @test scitype(dataset.W₂) == AbstractVector{Union{Missing, OrderedFactor{1}}}
    
    # Binary Outcome
    Ψ = IATE(
        outcome=:Ybin,
        treatment_values=(T₂=(case="AC", control="CC"), ),
        treatment_confounders=(T₂=[:W₂],),
        outcome_extra_covariates=[:C]
    )
    TargetedEstimation.coerce_types!(dataset, Ψ)
    @test scitype(dataset.Ybin) == AbstractVector{Union{Missing, OrderedFactor{2}}}
    @test scitype(dataset.C) == AbstractVector{Count}

    # Count Outcome
    Ψ = IATE(
        outcome=:Ycount,
        treatment_values=(T₂=(case="AC", control="CC"), ),
        treatment_confounders=(T₂=[:W₂],),
    )
    TargetedEstimation.coerce_types!(dataset, Ψ)
    @test scitype(dataset.Ycount) == AbstractVector{Union{Missing, MLJBase.Continuous}}
end

@testset "Test misc" begin
    Ψ = ATE(
        outcome = :Y,
        treatment_values = (
            T₁ = (case=1, control=0), 
            T₂ = (case=1, control=0)),
        treatment_confounders = (
            T₁=[:W₁, :W₂], 
            T₂=[:W₂, :W₃]
        ),
        outcome_extra_covariates = [:C]
    )
    variables = TargetedEstimation.variables(Ψ)
    @test variables == Set([:Y, :C, :T₁, :T₂, :W₁, :W₂, :W₃])
    Ψ = ATE(
        outcome = :Y,
        treatment_values = (
            T₁ = (case=1, control=0), 
            T₂ = (case=1, control=0)),
        treatment_confounders = (
            T₁=[:W₁, :W₂], 
            T₂=[:W₁, :W₂]
        ),
    )
    variables = TargetedEstimation.variables(Ψ)
    @test variables == Set([:Y, :T₁, :T₂, :W₁, :W₂])
    data = DataFrame(
        SAMPLE_ID  = [1, 2, 3, 4, 5],
        Y          = [1, 2, 3, missing, 5],
        W₁         = [1, 2, 3, 4, 5],
        W₂         = [missing, 2, 3, 4, 5],
        T₁         = [1, 2, 3, 4, 5],
        T₂         = [1, 2, 3, 4, missing],
    )
    sample_ids = TargetedEstimation.sample_ids_from_variables(data, variables)
    @test sample_ids == [2, 3]
    data.W₁ = [1, 2, missing, 4, 5]
    sample_ids = TargetedEstimation.sample_ids_from_variables(data, variables)
    @test sample_ids == [2]
    # wrapped_ype
    col = categorical(["AC", "CC"])
    @test TargetedEstimation.wrapped_type(eltype(col)) == String
    col = categorical(["AC", "CC", missing])
    @test TargetedEstimation.wrapped_type(eltype(col)) == String
    col = [1, missing, 0.3]
    @test TargetedEstimation.wrapped_type(eltype(col)) == Float64
    col = [1, 2, 3]
    @test TargetedEstimation.wrapped_type(eltype(col)) == Int64

end

@testset "Test make_categorical! and make_float!" begin
    dataset = DataFrame(
        T₁ = [1, 1, 0, 0],
        T₂ = ["AA", "AC", "CC", "CC"],
    )
    TargetedEstimation.make_categorical!(dataset, (:T₁, :T₂))
    @test dataset.T₁ isa CategoricalVector
    @test dataset.T₁.pool.ordered == false
    @test dataset.T₂ isa CategoricalVector
    @test dataset.T₂.pool.ordered == false

    dataset = DataFrame(
        T₁ = [1, 1, 0, 0],
        T₂ = ["AA", "AC", "CC", "CC"],
        C₁ = [1, 2, 3, 4],
    )
    TargetedEstimation.make_categorical!(dataset, (:T₁, :T₂), infer_ordered=true)
    @test dataset.T₁ isa CategoricalVector
    @test dataset.T₁.pool.ordered == true
    @test dataset.T₂ isa CategoricalVector
    @test dataset.T₂.pool.ordered == false

    TargetedEstimation.make_float!(dataset, [:C₁])
    @test eltype(dataset.C₁) == Float64

    # If the type is already coerced then no-operation is applied 
    TargetedEstimation.make_float(dataset.C₁) === dataset.C₁
    TargetedEstimation.make_categorical(dataset.T₁, true) === dataset.T₁
end

end;

true