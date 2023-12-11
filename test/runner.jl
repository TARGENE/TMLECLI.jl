module TestsTMLE

using Test
using TargetedEstimation
using TMLE
using JLD2
using CSV
using Serialization
using YAML
using JSON

TESTDIR = joinpath(pkgdir(TargetedEstimation), "test")

CONFIGDIR = joinpath(TESTDIR, "config")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Integration Test" begin
    build_dataset(;n=1000, format="csv")
    tmpdir = mktempdir(cleanup=true)
    estimands_filename = joinpath(tmpdir, "configuration.yaml")
    TMLE.write_json(estimands_filename, statistical_estimands_only_config())
    outputs = TargetedEstimation.Outputs(
        json=TargetedEstimation.JSONOutput(filename="output.json"),
        hdf5=TargetedEstimation.HDF5Output(filename="output.hdf5", pval_threshold=1., sample_ids=true),
        jls=TargetedEstimation.JLSOutput(filename="output.jls", pval_threshold=1e-5),
    )
    runner = Runner(
        "data.csv";
        estimands=estimands_filename, 
        estimators=joinpath(CONFIGDIR, "tmle_ose_config.jl"),
        outputs=outputs, 
        cache_strategy="release-unusable",
    )
    partition = 4:6
    results = runner(partition)
    for result in results
        @test result.TMLE isa TMLE.TMLEstimate
        @test result.OSE isa TMLE.OSEstimate
    end

    # Save outputs
    TargetedEstimation.initialize(outputs)
    TargetedEstimation.save(runner, results, partition, true)

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
    for (index, (result, loaded_result)) in enumerate(zip(results, loaded_results))
        @test loaded_result[:TMLE] isa TMLE.TMLEstimate
        @test result.TMLE.estimate == loaded_result[:TMLE].estimate
        @test loaded_result[:OSE] isa TMLE.OSEstimate
        @test result.OSE.estimate == loaded_result[:OSE].estimate
        @test !haskey(loaded_result, :SAMPLE_IDS)
        if index ∈ (1, 2)
            @test loaded_result[:TMLE].IC == []
            @test loaded_result[:OSE].IC == []
        else
            @test length(loaded_result[:TMLE].IC) > 0
            @test length(loaded_result[:OSE].IC) > 0
        end
    end

    # Test Save to HDF5
    hdf5file = jldopen(outputs.hdf5.filename, "r")
    loaded_results = hdf5file[string("Batch_1")]
    for (param_index, result) in enumerate(loaded_results)
        @test result.TMLE isa TMLE.TMLEstimate
        @test results[param_index].TMLE.estimate == result.TMLE.estimate

        @test result.OSE isa TMLE.OSEstimate
        @test results[param_index].OSE.estimate == result.OSE.estimate
    end

    @test loaded_results[1].SAMPLE_IDS == collect(2:1000)
    @test size(loaded_results[1].TMLE.IC, 1) == 999

    @test loaded_results[2].SAMPLE_IDS == 1
    @test size(loaded_results[2].TMLE.IC, 1) == 999

    @test loaded_results[3].SAMPLE_IDS == collect(1:1000)
    @test size(loaded_results[3].TMLE.IC, 1) == 1000

    close(hdf5file)

    # Clean
    rm("data.csv")
    rm(outputs.jls.filename)
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
            tmle(datafile; 
                estimands=estimands_filename, 
                estimators=estimatorfile,
                outputs=outputs,
                chunksize=chunksize,
            )

            results_from_hdf5 = jldopen(outputs.hdf5.filename) do io
                map(keys(io)) do key
                    io[key]
                end
            end
            results_from_hdf5 = vcat(results_from_hdf5...)
            results_from_json = TMLE.read_json(outputs.json.filename)

            for i in 1:6
                Ψ = configuration.estimands[i]
                for estimator_name in (:OSE, :TMLE)
                    @test Ψ == results_from_hdf5[i][estimator_name].estimand == results_from_json[i][estimator_name].estimand
                    @test results_from_hdf5[i][estimator_name].estimate == results_from_json[i][estimator_name].estimate
                end
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
    tmle(datafile; 
        estimands=estimandsfile, 
        estimators=estimatorfile,
        outputs=outputs)
    
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

    runner = Runner(datafile; 
        estimands=estimandsfile, 
        estimators=estimatorfile,
        outputs=outputs
    );
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
    results_from_hdf5 = jldopen(outputs.hdf5.filename)["Batch_1"]
    for estimator in (:OSE, :TMLE)
        @test results_from_hdf5[1][estimator] isa TargetedEstimation.FailedEstimate
        @test results_from_hdf5[2][estimator] isa TMLE.EICEstimate
        for i in 3:6
            @test results_from_hdf5[i][estimator] isa TargetedEstimation.FailedEstimate
            @test results_from_hdf5[i][estimator].estimand isa TMLE.Estimand
        end
    end

    # Clean
    rm(outputs.json.filename)
    rm(outputs.hdf5.filename)
    rm(datafile)
end

@testset "Test tmle: Causal and Composed Estimands" begin
    build_dataset(;n=1000, format="csv")
    outputs = TargetedEstimation.Outputs(
        json = TargetedEstimation.JSONOutput(filename="output.json"),
        jls = TargetedEstimation.JLSOutput(filename="output.jls"),
        hdf5 = TargetedEstimation.HDF5Output(filename="output.hdf5")
    )
    tmpdir = mktempdir(cleanup=true)
    estimandsfile = joinpath(tmpdir, "configuration.jls")

    configuration = causal_and_composed_estimands_config()
    serialize(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")
    datafile = "data.csv"

    tmle(datafile;
        estimands=estimandsfile, 
        estimators=estimatorfile,
        outputs=outputs, 
        chunksize=2
    )
    
    # JLS Output
    results = []
    open(outputs.jls.filename) do io
        while !eof(io)
            push!(results, deserialize(io))
        end
    end
    for (index, Ψ) ∈ enumerate(configuration.estimands)
        @test results[index].OSE.estimand == identify(Ψ, configuration.scm)
    end
    # The components of the diff should match the estimands 1 and 2
    for index in 1:2
        ATE_from_diff = results[3].OSE.estimates[index] 
        ATE_standalone = results[index].OSE
        @test ATE_from_diff.estimand == ATE_standalone.estimand
        @test ATE_from_diff.estimate == ATE_standalone.estimate
        @test ATE_from_diff.std == ATE_standalone.std
    end
    @test results[3].OSE isa TMLE.ComposedEstimate
    
    # JSON Output
    results_from_json = TMLE.read_json(outputs.json.filename)
    @test length(results_from_json) == 3

    # HDF5
    results_from_json = jldopen(outputs.hdf5.filename)
    @test length(results_from_json["Batch_1"]) == 2
    composed_result = only(results_from_json["Batch_2"])
    @test composed_result.OSE.cov == results[3].OSE.cov
    
    rm(datafile)
    rm(outputs.jls.filename)
    rm(outputs.json.filename)
    rm(outputs.hdf5.filename)
end


end;

true