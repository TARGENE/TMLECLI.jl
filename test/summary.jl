module TestSummary

using TargetedEstimation
using Test
using CSV
using DataFrames
using Serialization
using JLD2

TESTDIR = joinpath(pkgdir(TargetedEstimation), "test")

CONFIGDIR = joinpath(TESTDIR, "config")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test make_summary" begin
    tmpdir = mktempdir()
    datafile = joinpath(tmpdir, "data.csv")
    write_dataset(datafile)
    
    estimatorfile = "wtmle--glmnet"
    # First Run
    tmle_output_1 = TargetedEstimation.Outputs(hdf5=joinpath(tmpdir, "tmle_output_1.hdf5"))
    config_1 = statistical_estimands_only_config()
    configfile_1 = joinpath(tmpdir, "configuration_1.json")
    TMLE.write_json(configfile_1, config_1)
    tmle(datafile; 
        estimands=configfile_1, 
        estimators=estimatorfile,
        outputs=tmle_output_1, 
        chunksize=3
    )
    
    # Second Run
    tmle_output_2 = TargetedEstimation.Outputs(hdf5=joinpath(tmpdir, "tmle_output_2.hdf5"))
    config_2 = causal_and_joint_estimands_config()
    configfile_2 = joinpath(tmpdir, "configuration_2.json")
    TMLE.write_json(configfile_2, config_2)
    tmle(datafile; 
        estimands=configfile_2, 
        estimators=estimatorfile, 
        outputs=tmle_output_2
    )

    # Using the main entry point
    json_output = joinpath(tmpdir, "summary.json")
    jls_output = joinpath(tmpdir, "summary.jls")
    hdf5_output = joinpath(tmpdir, "summary.hdf5")
    main([
        "merge", 
        joinpath(tmpdir, "tmle_output"), 
        "--json-output", json_output, 
        "--jls-output", jls_output,
        "--hdf5-output", hdf5_output
    ])

    # Test correctness
    inputs = TargetedEstimation.read_results_from_files([joinpath(tmpdir, "tmle_output_1.hdf5"), joinpath(tmpdir, "tmle_output_2.hdf5")])

    json_outputs = TMLE.read_json(json_output, use_mmap=false)
    jls_outputs = deserialize(jls_output)
    hdf5_outputs = jldopen(io -> io["results"], hdf5_output)

    @test length(inputs) == 9
    for (input, jls_output, hdf5_out, json_output) in zip(inputs, jls_outputs, hdf5_outputs, json_outputs)
        @test input.WTMLE.estimand == jls_output.WTMLE.estimand == hdf5_out.WTMLE.estimand == json_output[:WTMLE].estimand
    end
end


end

true