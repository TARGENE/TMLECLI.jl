module TestsUKBB

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

function test_parameters(params, expected_params)
    for (param, expected_param) in zip(params, expected_params)
        @test param.target == expected_param.target
        @test param.treatment == expected_param.treatment
        @test param.confounders == expected_param.confounders
        @test param.covariates == expected_param.covariates
        @test typeof(param) == typeof(expected_param)
    end
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

    CSV.write("data.csv", dataset)
end


@testset "Test tmle_run with: no extra covariate, influence curve no threshold, super learning" begin
    build_dataset(;n=1000, format="csv")
    # Only one continuous phenotype / machines not saved / no adaptive cv
    parsed_args = Dict(
        "data" => "data.csv",
        "param-file" => joinpath("config", "parameters_no_extra_covariate.yaml"),
        "estimator-file" => joinpath("config", "tmle_config.yaml"),
        "outprefix" => "output",
        "verbosity" => 0,
        "save-ic" => true,
        "pval-threshold" => 1.
    )

    main(parsed_args)

    ## Check HDF5 file
    hdf5file = string(parsed_args["outprefix"], ".hdf5")
    io = jldopen(hdf5file)
    # Parameters are saved only for the first target to save memory
    expected_params = [
        IATE(
            Symbol("CONTINUOUS, TARGET"), 
            (T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)), 
            [:W1, :W2], 
            Symbol[]
        ),
        IATE(
            Symbol("CONTINUOUS, TARGET"),  
            (T2 = (case = 0, control = 1), T1 = (case = 1, control = 0)),
            [:W1, :W2],
            Symbol[]
        ),
        ATE(
            Symbol("CONTINUOUS, TARGET"), 
            (T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)),
            [:W1, :W2], 
            Symbol[]
        )
    ]
    test_parameters(io["parameters"], expected_params)
    
    # results for CONTINUOUS_TARGET
    continuous_results = io["results"]["CONTINUOUS, TARGET"]
    @test continuous_results["sample_ids"] == 1:1000
    tmles = continuous_results["tmle_results"]
    @test pvalue(OneSampleTTest(tmles[1], 0.5)) > 0.05
    @test pvalue(OneSampleTTest(tmles[2], -0.5)) > 0.05
    @test pvalue(OneSampleTTest(tmles[3], -0.5)) > 0.05
    @test continuous_results["initial_estimates"] isa Vector{Union{Missing, Float64}}
    @test size(continuous_results["initial_estimates"], 1) == 3

    # results for BINARY_TARGET
    binary_results = io["results"]["BINARY_OR_TARGET"]
    @test binary_results["sample_ids"] == 2:1000
    tmles = binary_results["tmle_results"]
    for i in 1:3
        @test size(tmles[i].IC, 1) == 999
        @test TMLE.estimate(tmles[i]) isa Float64
    end
    @test binary_results["initial_estimates"] isa Vector{Union{Missing, Float64}}
    @test size(binary_results["initial_estimates"], 1) == 3

    close(io)

    ## Check CSV file
    csvfile = string(parsed_args["outprefix"], ".csv")
    data = CSV.read(csvfile, DataFrame)
    names(data)
    @test data.PARAMETER_TYPE == ["IATE", "IATE", "ATE", "IATE", "IATE", "ATE"]
    @test data.TARGET == ["CONTINUOUS, TARGET", "CONTINUOUS, TARGET", "CONTINUOUS, TARGET",
                        "BINARY/TARGET", "BINARY/TARGET", "BINARY/TARGET"]
    @test data.TREATMENTS == fill("T2_&_T1", 6)
    @test data.CONFOUNDERS == fill("W1_&_W2", 6)
    @test data.CASE == ["1_&_1", "0_&_1", "1_&_1", "1_&_1", "0_&_1", "1_&_1"]
    @test data.CONTROL == ["0_&_0", "1_&_0", "0_&_0", "0_&_0", "1_&_0", "0_&_0"]

    
    for col in [:INITIAL_ESTIMATE, :ESTIMATE, :STD, :PVALUE, :LWB, :UPB]
        @test data[!, col] isa Vector{Float64}
    end

    # Clean
    rm(csvfile)
    rm(hdf5file)
    rm(parsed_args["data"])
end

@testset "Test tmle_run with: no extra covariate, influence curve 0.01 threshold, super learning" begin
    build_dataset(;n=1000, format="csv")
    # Only one continuous phenotype / machines not saved / no adaptive cv
    parsed_args = Dict(
        "data" => "data.csv",
        "param-file" => joinpath("config", "parameters_no_extra_covariate.yaml"),
        "estimator-file" => joinpath("config", "tmle_config.yaml"),
        "outprefix" => "output",
        "verbosity" => 0,
        "save-ic" => true,
        "pval-threshold" => 0.01
    )

    main(parsed_args)

    ## Check HDF5 file
    hdf5file = string(parsed_args["outprefix"], ".hdf5")
    io = jldopen(hdf5file)
    continuous = io["results"]["CONTINUOUS, TARGET"]
    @test size(continuous["tmle_results"], 1) == 2
    @test size(continuous["logs"], 1) == 2
    @test size(continuous["initial_estimates"], 1) == 2

    @test !haskey(io["results"], "BINARY/TARGET")

    ## Check CSV file
    csvfile = string(parsed_args["outprefix"], ".csv")
    data = CSV.read(csvfile, DataFrame)
    @test size(data) == (6, 14)

    # Clean
    rm(csvfile)
    rm(hdf5file)
    rm(parsed_args["data"])

end


@testset "Test tmle_run with: extra covariate, no influence curve, classifier simple models" begin
    build_dataset(;n=1000, format="csv")
    parsed_args = Dict(
        "data" => "data.csv",
        "param-file" => joinpath("config", "parameters_extra_covariate.yaml"),
        "estimator-file" => joinpath("config", "tmle_config_2.yaml"),
        "outprefix" => "output",
        "verbosity" => 0,
        "save-ic" => false,
        "pval-threshold" => 1.
    )

    main(parsed_args)
    
    # Essential results
    out = CSV.read(string(parsed_args["outprefix"], ".csv"), DataFrame)
    some_expected_col_values = DataFrame(
        PARAMETER_TYPE=["IATE", "IATE", "ATE", "IATE", "IATE", "ATE"], 
        TREATMENTS=["T2_&_T1", "T2_&_T1", "T2_&_T1", "T2_&_T1", "T2_&_T1", "T2_&_T1"], 
        CASE=["1_&_1", "0_&_1", "1_&_1", "1_&_1", "0_&_1", "1_&_1"],
        CONTROL=["0_&_0", "1_&_0", "0_&_0", "0_&_0", "1_&_0", "0_&_0"], 
        TARGET=["CONTINUOUS, TARGET", "CONTINUOUS, TARGET", "CONTINUOUS, TARGET", "BINARY/TARGET", "BINARY/TARGET", "BINARY/TARGET"], 
        CONFOUNDERS=["W1_&_W2", "W1_&_W2", "W1_&_W2", "W1_&_W2", "W1_&_W2", "W1_&_W2"], 
        COVARIATES=["C1", "C1", "C1", "C1", "C1", "C1"]
    )
    @test some_expected_col_values ==
        out[!, [:PARAMETER_TYPE, :TREATMENTS, :CASE, :CONTROL, :TARGET, :CONFOUNDERS, :COVARIATES]]
    for colname in [:INITIAL_ESTIMATE, :ESTIMATE, :STD, :PVALUE, :LWB, :UPB]
        @test eltype(out[!, colname]) == Float64
    end

    rm(parsed_args["data"])
    rm(string(parsed_args["outprefix"], ".csv"))
end

@testset "Test tmle_run with: extra covariate, influence curve but none passes threshold" begin
    # Nop HDF5 file should be output if no parameter makes the threhsold
    build_dataset(;n=1000, format="csv")
    parsed_args = Dict(
        "data" => "data.csv",
        "param-file" => joinpath("config", "parameters_extra_covariate.yaml"),
        "estimator-file" => joinpath("config", "tmle_config_2.yaml"),
        "outprefix" => "output",
        "verbosity" => 0,
        "save-ic" => true,
        "pval-threshold" => -1
    )

    main(parsed_args)
    
    @test !isfile(string(parsed_args["outprefix"], ".hdf5"))

    rm(parsed_args["data"])
    rm(string(parsed_args["outprefix"], ".csv"))
end


end;

true