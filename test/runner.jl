module TestsTMLE

using Test
using TargetedEstimation
using TMLE
using JLD2
using CSV
using Serialization
using YAML
using JSON
using MLJBase
using MLJModels

PKGDIR = pkgdir(TargetedEstimation)
TESTDIR = joinpath(PKGDIR, "test")
CONFIGDIR = joinpath(TESTDIR, "config")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test instantiate_estimators" begin
    # From template name
    for file in readdir(joinpath(PKGDIR, "estimators-configs"))
        configname = replace(file, ".jl" => "")
        estimators = TargetedEstimation.instantiate_estimators(configname)
        @test estimators.TMLE isa TMLEE
    end
    # From explicit file
    estimators = TargetedEstimation.instantiate_estimators(joinpath(TESTDIR, "config", "tmle_ose_config.jl"))
    @test estimators.TMLE isa TMLE.TMLEE
    @test estimators.OSE isa TMLE.OSE
    @test estimators.TMLE.weighted === true
    @test estimators.TMLE.models.G_default === estimators.OSE.models.G_default
    @test estimators.TMLE.models.G_default.continuous_encoder isa MLJModels.ContinuousEncoder
    @test estimators.TMLE.models.G_default.probabilistic_stack isa MLJBase.ProbabilisticStack
    # From already constructed estimators
    estimators_new = TargetedEstimation.instantiate_estimators(estimators)
    @test estimators_new === estimators
end

@testset "Integration Test" begin
    dataset = build_dataset(;n=1000)
    tmpdir = mktempdir(cleanup=true)
    config = statistical_estimands_only_config()
    outputs = TargetedEstimation.Outputs(
        json=TargetedEstimation.JSONOutput(filename="output.json"),
        hdf5=TargetedEstimation.HDF5Output(filename="output.hdf5", pval_threshold=1., sample_ids=true),
        jls=TargetedEstimation.JLSOutput(filename="output.jls", pval_threshold=1e-5),
    )
    estimators = TargetedEstimation.instantiate_estimators(joinpath(CONFIGDIR, "tmle_ose_config.jl"))
    runner = Runner(
        dataset;
        estimands_config=config, 
        estimators_spec=estimators,
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
    loaded_results = TMLE.read_json(outputs.json.filename, use_mmap=false)
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
        write_dataset(;n=1000, format=format)
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
            results_from_json = TMLE.read_json(outputs.json.filename, use_mmap=false)

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
        GC.gc() # memory freed for deleting arrow file
        rm(datafile)
    end
end

@testset "Test tmle: lower p-value threshold only JSON output" begin
    write_dataset(;n=1000, format="csv")
    tmpdir = mktempdir(cleanup=true)
    estimandsfile = joinpath(tmpdir, "configuration.json")
    configuration = statistical_estimands_only_config()
    TMLE.write_json(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")
    datafile = "data.csv"

    # Using the main entry point
    main([
        "tmle", 
        datafile, 
        "--estimands", estimandsfile, 
        "--estimators", estimatorfile,
        "--json-output", "output.json,1e-15"]
    )
    
    # Essential results
    results_from_json = TMLE.read_json("output.json", use_mmap=false)
    n_IC_empties = 0
    for result in results_from_json
        if result[:OSE].IC != []
            n_IC_empties += 1
        end
    end
    @test n_IC_empties > 0

    rm(datafile)
    rm("output.json")
end

@testset "Test tmle: Failing estimands" begin
    write_dataset(;n=1000, format="csv")
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
        estimands_config=estimandsfile, 
        estimators_spec=estimatorfile,
        outputs=outputs
    );
    runner()

    # Test failed nuisance estimates (T2 model)
    @test runner.failed_nuisance == Set([
        TMLE.ConditionalDistribution(:T2, (:W1, :W2))
    ])

    # Check results from JSON
    results_from_json = TMLE.read_json(outputs.json.filename, use_mmap=false)
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
    jldopen(outputs.hdf5.filename) do io 
        results_from_hdf5 = io["Batch_1"]
        for estimator in (:OSE, :TMLE)
            @test results_from_hdf5[1][estimator] isa TargetedEstimation.FailedEstimate
            @test results_from_hdf5[2][estimator] isa TMLE.EICEstimate
            for i in 3:6
                @test results_from_hdf5[i][estimator] isa TargetedEstimation.FailedEstimate
                @test results_from_hdf5[i][estimator].estimand isa TMLE.Estimand
            end
        end
    end
    # Clean
    rm(outputs.json.filename)
    rm(outputs.hdf5.filename)
    rm(datafile)
end

@testset "Test tmle: Causal and Joint Estimands" begin
    write_dataset(;n=1000, format="csv")
    tmpdir = mktempdir(cleanup=true)
    estimandsfile = joinpath(tmpdir, "configuration.jls")

    configuration = causal_and_joint_estimands_config()
    serialize(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")
    datafile = "data.csv"

    # Using the main entry point
    main([
        "tmle", 
        datafile, 
        "--estimands", estimandsfile, 
        "--estimators", estimatorfile,
        "--chunksize", "2",
        "--json-output", "output.json",
        "--hdf5-output", "output.hdf5",
        "--jls-output", "output.jls"
    ])
    
    # JLS Output
    results = []
    open("output.jls") do io
        while !eof(io)
            push!(results, deserialize(io))
        end
    end
    for (index, Ψ) ∈ enumerate(configuration.estimands)
        @test results[index].OSE.estimand == identify(Ψ, configuration.scm)
    end
    # The components of the diff should match the estimands 1 and 2
    for index in 1:2
        ATE_from_joint = results[3].OSE.estimates[index] 
        ATE_standalone = results[index].OSE
        @test ATE_from_joint.estimand == ATE_standalone.estimand
        @test ATE_from_joint.estimate == ATE_standalone.estimate
        @test ATE_from_joint.std == ATE_standalone.std
    end
    @test results[3].OSE isa TMLE.JointEstimate
    
    # JSON Output
    results_from_json = TMLE.read_json("output.json", use_mmap=false)
    @test length(results_from_json) == 3

    # HDF5
    jldopen("output.hdf5") do io
        @test length(io["Batch_1"]) == 2
        jointresult = only(io["Batch_2"])
        @test jointresult.OSE.cov == results[3].OSE.cov
    end

    rm(datafile)
    rm("output.jls")
    rm("output.json")
    rm("output.hdf5")
end


end;

true