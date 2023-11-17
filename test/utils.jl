module TestUtils

using Test
using TargetedEstimation
using TMLE
using DataFrames
using CSV
using MLJBase
using MLJLinearModels
using CategoricalArrays

PROJECT_DIR = dirname(dirname(pathof(TargetedEstimation)))

include(joinpath(PROJECT_DIR, "test", "testutils.jl"))

@testset "Test load_tmle_spec: with configuration file" begin
    estimators = TargetedEstimation.load_tmle_spec(joinpath(PROJECT_DIR, "test", "config", "tmle_config.jl"))
    @test estimators.TMLE isa TMLE.TMLEE
    @test estimators.OSE isa TMLE.OSE
    @test estimators.TMLE.weighted === true
    @test estimators.TMLE.models.G_default === estimators.OSE.models.G_default
    @test estimators.TMLE.models.G_default isa MLJBase.ProbabilisticStack
end

@testset "Test load_tmle_spec: no configuration file" begin
    estimators = TargetedEstimation.load_tmle_spec(nothing)
    @test !haskey(estimators, :OSE)
    @test haskey(estimators, :TMLE)
    @test estimators.TMLE.weighted === true
    @test estimators.TMLE.models.G_default isa LogisticClassifier
end

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

@testset "Test proofread_estimands_from_yaml" begin
    filename = "statistical_estimands.yml"
    configuration_to_yaml(filename, statistical_estimands_only_config())
    dataset = DataFrame(T1 = [1., 0.], T2=[true, false])
    estimands = TargetedEstimation.proofread_estimands_from_yaml(filename, dataset)
    for estimand in estimands
        if haskey(estimand.treatment_values, :T1)
            @test estimand.treatment_values.T1.case isa Float64
            @test estimand.treatment_values.T1.control isa Float64
        end
        if haskey(estimand.treatment_values, :T2)
            @test estimand.treatment_values.T2.case isa Bool
            @test estimand.treatment_values.T2.control isa Bool
        end
    end
    rm(filename)
end

@testset "Test CSV writing" begin
    Ψ = IATE(
        outcome=:Y,
        treatment_values=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        treatment_confounders=(T₁=[:W₁, :W₂], T₂=[:W₁, :W₂])
    )
    @test TargetedEstimation.covariates_string(Ψ) === missing
    @test TargetedEstimation.param_string(Ψ) == "IATE"
    @test TargetedEstimation.case_string(Ψ) == "1_&_AC"
    @test TargetedEstimation.control_string(Ψ) == "0_&_CC"
    @test TargetedEstimation.treatment_string(Ψ) == "T₁_&_T₂"
    @test TargetedEstimation.confounders_string(Ψ) == "W₁_&_W₂"

    Ψ = CM(
        target=:Y,
        treatment=(T₁=1, T₂="AC"),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )

    @test TargetedEstimation.covariates_string(Ψ) === "C₁"
    @test TargetedEstimation.param_string(Ψ) == "CM"
    @test TargetedEstimation.case_string(Ψ) == "1_&_AC"
    @test TargetedEstimation.control_string(Ψ) === missing
    @test TargetedEstimation.treatment_string(Ψ) == "T₁_&_T₂"
    @test TargetedEstimation.confounders_string(Ψ) == "W₁_&_W₂"
end

@testset "Test coerce_types!" begin
    Ψ = IATE(
        outcome=:Ycont,
        treatment_values=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        treatment_confounders=(T₁=[:W₁, :W₂], T₂=[:W₁, :W₂]),
    )

    dataset = DataFrame(
        Ycont  = [1.1, 2.2, missing],
        Ycat = [1., 0., missing],
        T₁ = [1, 0, missing],
        T₂ = [missing, "AC", "CC"],
        W₁ = [1., 0., 0.],
        W₂ = [missing, 0., 0.],
        C = [1, 2, 3]
    )
    TargetedEstimation.coerce_types!(dataset, Ψ)

    @test dataset.T₁ isa CategoricalArray
    @test dataset.T₂ isa CategoricalArray
    for var in [:W₁, :W₂, :Ycont]
        @test eltype(dataset[!, var]) <: Union{Missing, Float64}
    end

    Ψ = IATE(
        outcome=:Ycat,
        treatment_values=(T₂=(case="AC", control="CC"), ),
        treatment_confounders=(T₂=[:W₂],),
        outcome_extra_covariates=[:C]
    )
    TargetedEstimation.coerce_types!(dataset, Ψ)

    @test dataset.Ycat isa CategoricalArray
    @test eltype(dataset.C) <: Union{Missing, Float64}

end

@testset "Test get_sample_ids" begin
    variables = (
        target = :Y,
        covariates = Symbol[],
        confounders = [:W₁, :W₂],
        treatments = (:T₁, :T₂)
    )
    data = DataFrame(
        SAMPLE_ID  = [1, 2, 3, 4, 5],
        Y          = [1, 2, 3, missing, 5],
        W₁         = [1, 2, 3, 4, 5],
        W₂         = [missing, 2, 3, 4, 5],
        T₁         = [1, 2, 3, 4, 5],
        T₂         = [1, 2, 3, 4, missing],
    )
    sample_ids = TargetedEstimation.get_sample_ids(data, variables)
    @test sample_ids == [2, 3]
    data.W₁ = [1, 2, missing, 4, 5]
    sample_ids = TargetedEstimation.get_sample_ids(data, variables)
    @test sample_ids == [2]
end

@testset "Test write_target_results with missing values" begin
    filename = "test.csv"
    parameters = [
        CM(
        target=:Y,
        treatment=(T₁=1, T₂="AC"),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )]
    tmle_results = [TargetedEstimation.FailedEstimation(parameters[1])]
    logs = ["Error X"]
    TargetedEstimation.append_csv(filename, tmle_results, logs)
    out = CSV.read(filename, DataFrame)
    expected_out = ["CM", "T₁_&_T₂", "1_&_AC", missing, "Y", "W₁_&_W₂", "C₁", 
        missing, missing, missing, missing, missing, missing,
        missing, missing, missing, missing, missing,
        "Error X"]
    for (x, y) in zip(first(out), expected_out)
        if x === missing 
            @test x === y
        else
            @test x == y
        end
    end
    rm(filename)
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