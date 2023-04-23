module TestUtils

using Test
using TargetedEstimation
using MLJ
using TMLE
using DataFrames
using CSV
using MLJBase
using CategoricalArrays

@testset "Test AdaptiveCV" begin
    # Continuous target
    cv = TargetedEstimation.AdaptiveCV(CV())
    n_sample_to_nfolds = ((10, 10), (200, 20), (1000, 10), (5000, 5), (20000, 3))
    for (n, expected_nfolds) in n_sample_to_nfolds
        y = rand(n)
        ttp = MLJBase.train_test_pairs(cv, 1:n, y)
        @test length(ttp) == expected_nfolds
        @test ttp == MLJBase.train_test_pairs(CV(nfolds=expected_nfolds), 1:n, y)
    end
    # Categorical target
    cv = TargetedEstimation.AdaptiveCV(StratifiedCV(nfolds=2))
    y = categorical(["a", "a", "a", "b", "b", "c", "c"])
    @test TargetedEstimation.countuniques(y) == [3, 2, 2]
    ## neff < 30 => nfolds = 5*neff = 7
    ttp = MLJBase.train_test_pairs(cv, 1:7, y)
    @test length(ttp) == 7
    @test ttp == MLJBase.train_test_pairs(StratifiedCV(nfolds=7), 1:7, y)
    
    ## neff = 2500 => 10
    n = 50_500
    y = categorical(vcat(repeat([true], 50_000), repeat([false], 500)))
    @test TargetedEstimation.countuniques(y) == [50_000, 500]
    ttp = MLJBase.train_test_pairs(cv, 1:n, y)
    @test length(ttp)== 10
    @test ttp == MLJBase.train_test_pairs(StratifiedCV(nfolds=10), 1:n, y)
end

@testset "Test CSV writing" begin
    Ψ = IATE(
        target=:Y,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        confounders=[:W₁, :W₂]
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

@testset "Test variables" begin
    parameters = [
        IATE(
        target=:Y,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        confounders=[:W₁, :W₂]),
        CM(
        target=:Y₂,
        treatment=(T₁=1, T₃="AC"),
        confounders=[:W₃, :W₂],
        covariates=[:C₁])
    ]
    dataset = DataFrame(Y=[1.1, 2.2, missing], Y₂=[1, 0, missing])
    variables = TargetedEstimation.variables(parameters, dataset)
    @test variables == (
        treatments = Set([:T₃, :T₁, :T₂]),
        confounders = Set([:W₁, :W₃, :W₂]),
        covariates = Set([:C₁]),
        binarytargets = Set([:Y₂]),
        continuoustargets = Set([:Y])
    )

    variables = TargetedEstimation.variables(parameters[1])
    @test variables == (
        target = :Y,
        covariates = Symbol[],
        confounders = [:W₁, :W₂],
        treatments = (:T₁, :T₂)
    )
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

@testset "Test treatment_values" begin
    treatment_types = Dict(:T₁=> Union{Missing, Bool}, :T₂=> Int)
    Ψ = CM(target=:Y, treatment=(T₁=1,), confounders=[:W₁])
    newT = TargetedEstimation.treatment_values(Ψ, (:T₁,), treatment_types)
    @test newT isa Vector{Bool}
    @test newT == [1]

    Ψ = ATE(target=:Y, treatment=(T₁=(case=1, control=0.),), confounders=[:W₁])
    newT = TargetedEstimation.treatment_values(Ψ, (:T₁,), treatment_types)
    @test newT isa Vector{NamedTuple{(:case, :control), Tuple{Bool, Bool}}}
    @test newT == [(case = true, control = false)]

    Ψ = ATE(target=:Y, treatment=(T₁=(case=1, control=0.), T₂=(case=true, control=0)), confounders=[:W₁])
    newT = TargetedEstimation.treatment_values(Ψ, (:T₁, :T₂), treatment_types)
    @test newT isa Vector{NamedTuple{(:case, :control)}}
    @test newT == [(case = true, control = false), (case = 1, control = 0)]
end

@testset "Test read_parameters" for param_file in ("parameters.yaml", "parameters.bin")
    param_file = joinpath("config", param_file)
    dataset = DataFrame(T1 = [1., 0.], T2=[true, false])
    params = TargetedEstimation.read_parameters(param_file, dataset)
    for param in params
        if haskey(param.treatment, :T1)
            @test param.treatment.T1.case isa Float64
            @test param.treatment.T1.control isa Float64
        end
        if haskey(param.treatment, :T2)
            @test param.treatment.T2.case isa Bool
            @test param.treatment.T2.control isa Bool
        end
    end
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
    tmle_results = [TargetedEstimation.MissingTMLEResult(parameters[1])]
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

end

end;

true