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
using Serialization
using Arrow
using YAML

PKGDIR = pkgdir(TargetedEstimation)

CONFIGDIR = joinpath(PKGDIR, "test", "config")

include(joinpath(PKGDIR, "test", "testutils.jl"))

sort_nt_by_key(nt::NamedTuple{names}) where names = NamedTuple{sort(names)}(nt)
sort_nt_by_key(x) = x

function test_estimands_match(Ψ₁::T1, Ψ₂::T2) where {T1, T2}
    @test T1 == T2
    @test Ψ₁.outcome == Ψ₂.outcome
    @test Ψ₁.outcome_extra_covariates == Ψ₂.outcome_extra_covariates
    @test sort_nt_by_key(Ψ₁.treatment_confounders) == sort_nt_by_key(Ψ₂.treatment_confounders)
    @test sort(keys(Ψ₁.treatment_values)) == sort(keys(Ψ₂.treatment_values))
    for key in keys(Ψ₁.treatment_values)
        @test sort_nt_by_key(Ψ₁.treatment_values[key]) == sort_nt_by_key(Ψ₂.treatment_values[key])
    end
end

"""
CONTINUOUS_OUTCOME: 
- IATE(0->1, 0->1) = E[W₂] = 0.5
- ATE(0->1, 0->1)  = -4 E[C₁] + 1 + E[W₂] = -2 + 1 + 0.5 = -0.5

BINARY_OUTCOME:
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
    dataset[!, "CONTINUOUS, OUTCOME"] = categorical(y₁)
    # Slash in name
    dataset[!, "BINARY/OUTCOME"] = categorical(y₂)
    dataset[!, "EXTREME_BINARY"] = categorical(vcat(0, ones(n-1)))

    format == "csv" ? CSV.write("data.csv", dataset) : Arrow.write("data.arrow", dataset)
end

@testset "Integration Test" begin
    build_dataset(;n=1000, format="csv")
    tmpdir = mktempdir(cleanup=true)
    estimands_filename = joinpath(tmpdir, "configuration.yaml")
    TMLE.write_json(estimands_filename, statistical_estimands_only_config())
    outputs = TargetedEstimation.Outputs(
        json=TargetedEstimation.JSONOutput(filename="output.json"),
        hdf5=TargetedEstimation.HDF5Output(filename="output.hdf5", pval_threshold=1.),
        jls=TargetedEstimation.JLSOutput(filename="output.jls"),
        std=true,
    )
    runner = Runner(
        "data.csv", 
        estimands_filename, 
        joinpath(CONFIGDIR, "tmle_ose_config.jl"); 
        outputs=outputs, 
        cache_strategy="release-unusable",
    )
    partition = 4:6
    results = runner(partition)
    for result in results
        @test result.TMLE isa TMLE.TMLEstimate
        @test result.OSE isa TMLE.OSEstimate
    end

    # Test Save to STDOUT
    output_txt = "output.txt"
    TargetedEstimation.initialize(outputs)
    open(output_txt, "w") do io
        redirect_stdout(io) do
            TargetedEstimation.save(runner, results, partition, true)
        end
    end
    stdout_content = read(output_txt, String)
    @test all(occursin("Estimand $i", stdout_content) for i in partition)

    # Test Save to JSON
    loaded_results = TMLE.read_json(outputs.json.filename)
    for (result, loaded_result) in zip(results, loaded_results)
        @test loaded_result[:TMLE] isa TMLE.TMLEstimate
        @test result.TMLE.estimate == loaded_result[:TMLE].estimate
        @test loaded_result[:TMLE].IC == []

        @test loaded_result[:OSE] isa TMLE.OSEstimate
        @test result.OSE.estimate == loaded_result[:OSE].estimate
        @test loaded_result[:OSE].IC == []
    end

    # Test Save to JLS
    loaded_results = []
    open(outputs.jls.filename) do io
        while !eof(io)
            push!(loaded_results, deserialize(io))
        end
    end
    for (result, loaded_result) in zip(results, loaded_results)
        @test loaded_result[:TMLE] isa TMLE.TMLEstimate
        @test result.TMLE.estimate == loaded_result[:TMLE].estimate
        @test loaded_result[:TMLE].IC == []

        @test loaded_result[:OSE] isa TMLE.OSEstimate
        @test result.OSE.estimate == loaded_result[:OSE].estimate
        @test loaded_result[:OSE].IC == []
    end

    # Test Save to HDF5
    hdf5file = jldopen(outputs.hdf5.filename, "r")
    for (result_index, param_index) in enumerate(4:6)
        result = hdf5file[string(param_index, "/result")]
        @test result.TMLE isa TMLE.TMLEstimate
        @test results[result_index].TMLE.estimate == result.TMLE.estimate

        @test result.OSE isa TMLE.OSEstimate
        @test results[result_index].OSE.estimate == result.OSE.estimate
    end
    @test hdf5file["4/sample_ids"] == collect(2:1000)
    @test hdf5file["4/sample_ids_idx"] == 4
    @test size(hdf5file["4/result"].TMLE.IC, 1) == 999

    @test !haskey(hdf5file, "5/sample_ids")
    @test hdf5file["5/sample_ids_idx"] == 4
    @test size(hdf5file["5/result"].TMLE.IC, 1) == 999

    @test hdf5file["6/sample_ids"] == collect(1:1000)
    @test hdf5file["6/sample_ids_idx"] == 6
    @test size(hdf5file["6/result"].TMLE.IC, 1) == 1000

    close(hdf5file)

    # Clean
    rm("data.csv")
    rm(output_txt)
    rm(outputs.json.filename)
    rm(outputs.hdf5.filename)
end

@testset "Test tmle: varying dataset format and chunksize" begin
    tmpdir = mktempdir(cleanup=true)
    estimands_filename = joinpath(tmpdir, "configuration.json")
    configuration = statistical_estimands_only_config()
    TMLE.write_json(estimands_filename, configuration)
    outputs = TargetedEstimation.Outputs(
        json=TargetedEstimation.JSONOutput(filename="output.json"),
        hdf5=TargetedEstimation.HDF5Output(filename="output.hdf5", pval_threshold=1.),
    )
    estimatorfile = joinpath(CONFIGDIR, "tmle_ose_config.jl")
    # Run tests over CSV and Arrow data formats
    for format in ("csv", "arrow")
        datafile = string("data.", format)
        build_dataset(;n=1000, format=format)
        for chunksize in (4, 10)
            tmle(datafile, estimands_filename, estimatorfile; 
                outputs=outputs,
                chunksize=chunksize,
            )

            hdf5file = jldopen(outputs.hdf5.filename)
            results_from_json = TMLE.read_json(outputs.json.filename)

            for i in 1:6
                Ψ = configuration.estimands[i]
                test_estimands_match(Ψ, results_from_json[i][:TMLE].estimand)
                hdf5result = hdf5file[string(i, "/result")]
                @test results_from_json[i][:TMLE].estimate == hdf5result.TMLE.estimate
                @test results_from_json[i][:OSE].estimate == hdf5result.OSE.estimate
            end

            # Clean
            rm(outputs.hdf5.filename)
            rm(outputs.json.filename)
        end
        rm(datafile)
    end
end

@testset "Test tmle: lower p-value threshold only JSON output" begin
    build_dataset(;n=1000, format="csv")
    outputs = TargetedEstimation.Outputs(
        json=TargetedEstimation.JSONOutput(filename="output.json", pval_threshold=1e-15)
    )
    tmpdir = mktempdir(cleanup=true)
    estimandsfile = joinpath(tmpdir, "configuration.json")
    configuration = statistical_estimands_only_config()
    TMLE.write_json(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")
    datafile = "data.csv"
    tmle(datafile, estimandsfile, estimatorfile; outputs=outputs)
    
    # Essential results
    results_from_json = TMLE.read_json(outputs.json.filename)
    n_IC_empties = 0
    for result in results_from_json
        if result[:OSE].IC != []
            n_IC_empties += 1
        end
    end
    @test n_IC_empties > 0

    rm(datafile)
    rm(outputs.json.filename)
end

@testset "Test tmle: Failing estimands" begin
    build_dataset(;n=1000, format="csv")
    outputs = TargetedEstimation.Outputs(
        json=TargetedEstimation.JSONOutput(filename="output.json"),
        hdf5=TargetedEstimation.HDF5Output(filename="output.hdf5")
    )
    tmpdir = mktempdir(cleanup=true)
    estimandsfile = joinpath(tmpdir, "configuration.json")
    configuration = statistical_estimands_only_config()
    TMLE.write_json(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "problematic_tmle_ose_config.jl")
    datafile = "data.csv"

    runner = Runner(datafile, estimandsfile, estimatorfile; outputs=outputs);
    runner()

    # Test failed nuisance estimates (T2 model)
    @test runner.failed_nuisance == Set([
        TMLE.ConditionalDistribution(:T2, (:W1, :W2))
    ])

    # Check results from JSON
    results_from_json = TMLE.read_json(outputs.json.filename)
    for estimator in (:OSE, :TMLE)
        @test results_from_json[1][estimator][:error] == "Could not fit the following propensity score model: P₀(T2 | W1, W2)"
        @test results_from_json[1][estimator][:estimand] isa TMLE.Estimand
        @test results_from_json[2][estimator] isa TMLE.EICEstimate
        for i in 3:6
            @test results_from_json[i][estimator][:error] == "Skipped due to shared failed nuisance fit."
            @test results_from_json[i][estimator][:estimand] isa TMLE.Estimand
        end
    end

    # Check results from HDF5
    hdf5file = jldopen(outputs.hdf5.filename)
    for estimator in (:OSE, :TMLE)
        @test hdf5file["1/result"][estimator] isa TargetedEstimation.FailedEstimation
        @test hdf5file["2/result"][estimator] isa TMLE.EICEstimate
        for i in 3:6
            @test hdf5file[string(i, "/result")][estimator] isa TargetedEstimation.FailedEstimation
            @test hdf5file[string(i, "/result")][estimator].estimand isa TMLE.Estimand
        end
    end
    close(hdf5file)

    # Clean
    rm(outputs.json.filename)
    rm(outputs.hdf5.filename)
    rm(datafile)
end

@testset "Test tmle: Causal and Composed Estimands" begin
    build_dataset(;n=1000, format="csv")
    outputs = TargetedEstimation.Outputs(
        jls=TargetedEstimation.JLSOutput(filename="output.jls")
    )
    tmpdir = mktempdir(cleanup=true)
    estimandsfile = joinpath(tmpdir, "configuration.jls")

    configuration = causal_and_composed_estimands_config()
    serialize(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")
    datafile = "data.csv"
    tmle(datafile, estimandsfile, estimatorfile; outputs=outputs)
    
    rm(datafile)
end


end;

true