module TestsTMLE

using Test
using TargetedEstimation
using TMLE
using JLD2
using StableRNGs
using Distributions
using LogExpFunctions
using CategoricalArrays
using DataFrames
using CSV
using Arrow

function test_tmle_output(param_index, jldio, data, expected_param, sample_ids_idx)
    jld2_res = jldio[string(param_index)]
    csv_row = data[param_index, :]
    Ψ = jld2_res["result"].parameter
    @test jld2_res["result"] isa TMLE.TMLEResult
    @test jld2_res["result"].tmle.Ψ̂ isa Float64
    @test Ψ == expected_param
    @test jld2_res["sample_ids_idx"] == sample_ids_idx
    sample_ids = jldio[string(jld2_res["sample_ids_idx"])]["sample_ids"]
    if expected_param.target == Symbol("BINARY/TARGET")
        @test sample_ids == 2:1000
    else
        @test sample_ids == 1:1000
    end
    @test jld2_res["result"] isa TMLE.TMLEResult

    if csv_row.COVARIATES === missing
        @test TargetedEstimation.covariates_string(Ψ) === csv_row.COVARIATES
    else
        @test TargetedEstimation.covariates_string(Ψ) == csv_row.COVARIATES
    end
    @test TargetedEstimation.param_string(Ψ) == csv_row.PARAMETER_TYPE
    @test TargetedEstimation.case_string(Ψ) == csv_row.CASE
    @test TargetedEstimation.control_string(Ψ) == csv_row.CONTROL
    @test TargetedEstimation.treatment_string(Ψ) == csv_row.TREATMENTS
    @test TargetedEstimation.confounders_string(Ψ) == csv_row.CONFOUNDERS
    @test csv_row.TMLE_ESTIMATE == jld2_res["result"].tmle.Ψ̂
end

"""
CONTINUOUS_TARGET: 
- IATE(0->1, 0->1) = E[W₂] = 0.5
- ATE(0->1, 0->1)  = -4 E[C₁] + 1 + E[W₂] = -2 + 1 + 0.5 = -0.5

BINARY_TARGET:
- IATE(0->1, 0->1) =
- ATE(0->1, 0->1)  = 

"""
function build_dataset(;n=1000, format="csv")
    rng = StableRNG(123)
    # Confounders
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< logistic.(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< logistic.(-3W₁ - 1.5W₂)
    # target | Confounders, Covariates, Treatments
    μ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .+ T₁ + T₂.*W₂.*T₁
    y₁ = μ .+ rand(rng, Normal(0, 0.01), n)
    y₂ = rand(rng, Uniform(), n) .< logistic.(μ)
    # Add some missingness
    y₂ = vcat(missing, y₂[2:end])

    dataset = DataFrame(
        SAMPLE_ID = 1:n,
        T1 = categorical(T₁),
        T2 = categorical(T₂),
        W1 = W₁, 
        W2 = W₂,
        C1 = C₁,
    )
    # Comma in name
    dataset[!, "CONTINUOUS, TARGET"] = categorical(y₁)
    # Slash in name
    dataset[!, "BINARY/TARGET"] = categorical(y₂)
    dataset[!, "EXTREME_BINARY"] = categorical(vcat(0, ones(n-1)))

    format == "csv" ? CSV.write("data.csv", dataset) : Arrow.write("data.arrow", dataset)
end

@testset "Test partition_tmle!" begin
    build_dataset(;n=1000, format="csv")
    dataset = TargetedEstimation.instantiate_dataset("data.csv")
    parameters = TargetedEstimation.read_parameters(joinpath("config", "parameters.yaml"), dataset)
    variables = TargetedEstimation.variables(parameters, dataset)
    TargetedEstimation.coerce_types!(dataset, variables)
    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(joinpath("config", "tmle_config.yaml"))
    cache = TMLECache(dataset)

    tmle_results = Vector{Union{TMLE.TMLEResult, Missing}}(undef, 3)
    logs = Vector{Union{String, Missing}}(undef, 3)
    part = 4:6
    TargetedEstimation.partition_tmle!(cache, tmle_results, logs, part, tmle_spec, parameters, variables; verbosity=0)
    @test [x.tmle.Ψ̂ for x in tmle_results] isa Vector{Float64}
    @test [x.parameter for x in tmle_results] == parameters[part]
    @test [x.onestep.Ψ̂ for x in tmle_results] isa Vector{Float64}
    @test all(x === missing for x in logs)
    rm("data.csv")
end

@testset "Test tmle_estimation" begin
    expected_parameters = [
        ATE(Symbol("CONTINUOUS, TARGET"), (T1 = (case = true, control = false),), [:W1, :W2], Symbol[]),
        IATE(Symbol("BINARY/TARGET"), (T1 = (case = true, control = false), T2 = (case = true, control = false)), [:W1, :W2], [:C1]),
        IATE(Symbol("BINARY/TARGET"), (T1 = (case = true, control = false), T2 = (case = false, control = true)), [:W1, :W2], [:C1]),
        IATE(Symbol("CONTINUOUS, TARGET"), (T1 = (case = true, control = false), T2 = (case = false, control = true)), [:W1, :W2], Symbol[]),
        IATE(Symbol("CONTINUOUS, TARGET"), (T1 = (case = true, control = false), T2 = (case = true, control = false)), [:W1, :W2], [:C1]),
        ATE(Symbol("CONTINUOUS, TARGET"), (T1 = (case = true, control = false), T2 = (case = true, control = false)), [:W1, :W2], [:C1])
    ]
    expected_param_sample_ids_idx = [1, 2, 2, 4, 5, 5]
    # Run tests over CSV and Arrow data formats
    for format in ("csv", "arrow")
        build_dataset(;n=1000, format=format)
        parsed_args = Dict(
                    "data" => string("data.", format),
                    "param-file" => nothing,
                    "estimator-file" => joinpath("config", "tmle_config.yaml"),
                    "csv-out" => "output.csv",
                    "verbosity" => 0,
                    "hdf5-out" => "output.hdf5",
                    "pval-threshold" => 1.,
                    "chunksize" => nothing
                )
        for param_file in ("parameters.yaml", "parameters.bin")
            for chunksize in (4, 10)
                # Only one continuous phenotype / machines not saved / no adaptive cv
                parsed_args["param-file"] = joinpath("config", param_file)
                parsed_args["chunksize"] = chunksize

                tmle_estimation(parsed_args)

                # Given the threshold is 1, all
                # estimation results will make the threshold
                jldio = jldopen(parsed_args["hdf5-out"])
                data = CSV.read(parsed_args["csv-out"], DataFrame)

                @test all(data[i, :TMLE_ESTIMATE] != data[j, :TMLE_ESTIMATE] for i in 1:5 for j in i+1:6)

                for (param_index, (Ψ, sample_ids_idx)) in enumerate(zip(expected_parameters, expected_param_sample_ids_idx))
                    test_tmle_output(param_index, jldio, data, Ψ, sample_ids_idx)
                end
                # Clean
                rm(parsed_args["csv-out"])
                rm(parsed_args["hdf5-out"])
            end
        end
        rm(parsed_args["data"])
    end
end

@testset "Test tmle_estimation: No hdf5 file" begin
    build_dataset(;n=1000, format="csv")
    # Only one continuous phenotype / machines not saved / no adaptive cv
    param_file = "parameters.yaml"
    parsed_args = Dict(
        "data" => "data.csv",
        "param-file" => joinpath("config", param_file),
        "estimator-file" => joinpath("config", "tmle_config.yaml"),
        "csv-out" => "output.csv",
        "verbosity" => 0,
        "hdf5-out" => nothing,
        "pval-threshold" => 1.,
        "chunksize" => 10
    )

    tmle_estimation(parsed_args)

    ## Check CSV file
    data = CSV.read(parsed_args["csv-out"], DataFrame)
    @test names(TargetedEstimation.csv_headers()) == names(data)
    @test size(data) == (6, 19)
    all(x === missing for x in data.LOG)
    # Clean
    rm(parsed_args["csv-out"])
    rm(parsed_args["data"])
end


@testset "Test tmle_estimation: lower p-value threhsold" begin
    build_dataset(;n=1000, format="csv")
    parsed_args = Dict(
        "data" => "data.csv",
        "param-file" => joinpath("config", "parameters.yaml"),
        "estimator-file" => joinpath("config", "tmle_config.yaml"),
        "csv-out" => "output.csv",
        "verbosity" => 0,
        "hdf5-out" => "output.hdf5",
        "pval-threshold" => 1e-10,
        "chunksize" => 10
    )

    tmle_estimation(parsed_args)
    
    # Essential results
    data = CSV.read(parsed_args["csv-out"], DataFrame)
    jldio = jldopen(parsed_args["hdf5-out"])
    @test !haskey(jldio, "2")
    @test !haskey(jldio, "3")
    @test !haskey(jldio, "4")

    @test jldio["1"]["result"].tmle.Ψ̂ == data[1, :TMLE_ESTIMATE]
    @test jldio["5"]["result"].tmle.Ψ̂ == data[5, :TMLE_ESTIMATE]
    @test jldio["6"]["result"].tmle.Ψ̂ == data[6, :TMLE_ESTIMATE]

    rm(parsed_args["data"])
    rm(parsed_args["csv-out"])
    rm(parsed_args["hdf5-out"])
end

@testset "Test tmle_estimation: Failing parameters" begin
    build_dataset(;n=1000, format="csv")
    parsed_args = Dict(
        "data" => "data.csv",
        "param-file" => joinpath("config", "failing_parameters.yaml"),
        "estimator-file" => joinpath("config", "tmle_config.yaml"),
        "csv-out" => "output.csv",
        "verbosity" => 0,
        "hdf5-out" => nothing,
        "pval-threshold" => 1e-10,
        "chunksize" => 10
    )

    tmle_estimation(parsed_args)

    # Essential results
    data = CSV.read(parsed_args["csv-out"], DataFrame)
    @test size(data) == (1, 19)
    @test data[1, :TMLE_ESTIMATE] === missing

    rm(parsed_args["data"])
    rm(parsed_args["csv-out"])

end

end;

true