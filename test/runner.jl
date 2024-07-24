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

@testset "Test instantiate_estimators from file" begin
    # From explicit file
    estimators = TargetedEstimation.instantiate_estimators(joinpath(TESTDIR, "config", "tmle_ose_config.jl"), nothing)
    @test estimators.TMLE isa TMLE.TMLEE
    @test estimators.OSE isa TMLE.OSE
    @test estimators.TMLE.weighted === true
    @test estimators.TMLE.models.G_default === estimators.OSE.models.G_default
    @test estimators.TMLE.models.G_default.continuous_encoder isa MLJModels.ContinuousEncoder
    @test estimators.TMLE.models.G_default.probabilistic_stack isa MLJBase.ProbabilisticStack
    # From already constructed estimators
    estimators_new = TargetedEstimation.instantiate_estimators(estimators, nothing)
    @test estimators_new === estimators
end

@testset "Integration Test" begin
    tmpdir = mktempdir()
    dataset = build_dataset(;n=1000)
    
    config = statistical_estimands_only_config()
    outputs = TargetedEstimation.Outputs(
        json=joinpath(tmpdir, "output.json"),
        hdf5=joinpath(tmpdir, "output.hdf5"),
        jls=joinpath(tmpdir, "output.jls"),
    )
    estimators = TargetedEstimation.instantiate_estimators(joinpath(CONFIGDIR, "tmle_ose_config.jl"), nothing)
    runner = Runner(
        dataset;
        estimands_config=config, 
        estimators_spec=estimators,
        outputs=outputs, 
        cache_strategy="release-unusable",
        save_sample_ids=true,
        pvalue_threshold=1e-5
    )
    # Initialize outputs
    TargetedEstimation.initialize(outputs)
    # Run
    partition = 4:6
    results = runner(partition)
    for result in results
        @test result.TMLE isa TMLE.TMLEstimate
        @test result.OSE isa TMLE.OSEstimate
    end
    # Update outputs
    TargetedEstimation.update_outputs(runner, results)
    # Finalize outputs
    TargetedEstimation.finalize(runner.outputs)

    # Test Save to JSON
    loaded_json_results = TMLE.read_json(outputs.json, use_mmap=false)
    # Test Save to HDF5
    loaded_hdf5_results = jldopen(io -> io["Batch_1"], outputs.hdf5, "r")
    # Test Save to JLS
    loaded_jls_results = []
    open(outputs.jls) do io
        while !eof(io)
            push!(loaded_jls_results, deserialize(io))
        end
    end

    # Test loaded_results
    for loaded_results ∈ (loaded_json_results, loaded_hdf5_results, loaded_jls_results)
        # The pvalue-threshold is too stringent for estimands 1 and 2
        loaded_result_1 = loaded_results[1]
        @test loaded_result_1[:SAMPLE_IDS] == collect(2:1000)
        @test size(loaded_result_1[:TMLE].IC, 1) == size(loaded_result_1[:OSE].IC, 1) == 0

        loaded_result_2 = loaded_results[2]
        @test loaded_result_2[:SAMPLE_IDS] == 1
        @test size(loaded_result_2[:TMLE].IC, 1) == size(loaded_result_2[:OSE].IC, 1) == 0

        loaded_result_3 = loaded_results[3]
        @test loaded_result_3[:SAMPLE_IDS] == collect(1:1000)
        @test size(loaded_result_3[:TMLE].IC, 1) == size(loaded_result_3[:OSE].IC, 1) == 1000

        # Check loaded results match results
        for (result, loaded_result) in zip(results, loaded_results)
            @test loaded_result[:TMLE] isa TMLE.TMLEstimate
            @test result.TMLE.estimate == loaded_result[:TMLE].estimate

            @test loaded_result[:OSE] isa TMLE.OSEstimate
            @test result.OSE.estimate == loaded_result[:OSE].estimate
        end
    end
end

@testset "Test tmle: varying dataset format and chunksize" begin
    tmpdir = mktempdir()
    estimands_filename = joinpath(tmpdir, "configuration.json")
    configuration = statistical_estimands_only_config()
    TMLE.write_json(estimands_filename, configuration)
    outputs = TargetedEstimation.Outputs(
        json=joinpath(tmpdir, "output.json"),
        hdf5=joinpath(tmpdir, "output.hdf5"),
    )
    estimatorfile = joinpath(CONFIGDIR, "tmle_ose_config.jl")
    # Run tests over CSV and Arrow data formats
    for format in ("csv", "arrow")
        datafile = joinpath(tmpdir, string("data.", format))
        write_dataset(datafile; n=1000)
        for chunksize in (4, 10)
            tmle(datafile; 
                estimands=estimands_filename, 
                estimators=estimatorfile,
                outputs=outputs,
                chunksize=chunksize,
            )

            results_from_hdf5 = jldopen(outputs.hdf5) do io
                map(keys(io)) do key
                    io[key]
                end
            end
            results_from_hdf5 = vcat(results_from_hdf5...)
            results_from_json = TMLE.read_json(outputs.json, use_mmap=false)

            for i in 1:6
                Ψ = configuration.estimands[i]
                for estimator_name in (:OSE, :TMLE)
                    @test Ψ == results_from_hdf5[i][estimator_name].estimand == results_from_json[i][estimator_name].estimand
                    @test results_from_hdf5[i][estimator_name].estimate == results_from_json[i][estimator_name].estimate
                end
            end

        end
        GC.gc() # memory freed for deleting arrow file
    end
end

@testset "Test tmle: lower p-value threshold only JSON output" begin
    tmpdir = mktempdir()
    datafile = joinpath(tmpdir, "data.csv")
    write_dataset(datafile)
    estimandsfile = joinpath(tmpdir, "configuration.json")
    configuration = statistical_estimands_only_config()
    TMLE.write_json(estimandsfile, configuration)
    output = joinpath(tmpdir, "output.json")
    # Using the main entry point
    main([
        "tmle", 
        datafile, 
        "--estimands", estimandsfile, 
        "--estimators=ose--glm",
        "--pvalue-threshold=1e-15",
        "--json-output", output]
    )
    
    # Essential results
    results_from_json = TMLE.read_json(output, use_mmap=false)
    n_IC_empties = 0
    for result in results_from_json
        if result[:OSE].IC != []
            n_IC_empties += 1
        end
    end
    @test n_IC_empties > 0
end

@testset "Test tmle: Failing estimands" begin
    tmpdir = mktempdir()
    datafile = joinpath(tmpdir, "data.csv")
    write_dataset(datafile)
    outputs = TargetedEstimation.Outputs(
        json=joinpath(tmpdir, "output.json"),
        hdf5=joinpath(tmpdir, "output.hdf5")
    )
    estimandsfile = joinpath(tmpdir, "configuration.json")
    configuration = statistical_estimands_only_config()
    TMLE.write_json(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "problematic_tmle_ose_config.jl")

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
    results_from_json = TMLE.read_json(outputs.json, use_mmap=false)
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
    jldopen(outputs.hdf5) do io 
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
end

@testset "Test tmle: Causal and Joint Estimands" begin
    tmpdir = mktempdir()
    datafile = joinpath(tmpdir, "data.csv")
    write_dataset(datafile)
    
    estimandsfile = joinpath(tmpdir, "configuration.jls")

    configuration = causal_and_joint_estimands_config()
    serialize(estimandsfile, configuration)
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")

    jls_output = joinpath(tmpdir, "output.jls")
    hdf5_output = joinpath(tmpdir, "output.hdf5")
    json_output = joinpath(tmpdir, "output.json")
    # Using the main entry point
    main([
        "tmle", 
        datafile, 
        "--estimands", estimandsfile, 
        "--estimators", estimatorfile,
        "--chunksize", "2",
        "--json-output", json_output,
        "--hdf5-output", hdf5_output,
        "--jls-output", jls_output
    ])
    
    # JLS Output
    results = []
    open(jls_output) do io
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
    results_from_json = TMLE.read_json(json_output, use_mmap=false)
    @test length(results_from_json) == 3

    # HDF5
    jldopen(hdf5_output) do io
        @test length(io["Batch_1"]) == 2
        jointresult = only(io["Batch_2"])
        @test jointresult.OSE.cov == results[3].OSE.cov
    end
end


end;

true