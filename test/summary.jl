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
    write_dataset()
    datafile = "data.csv"
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")
    tmpdir = mktempdir(cleanup=true)
    # First Run
    tmle_output_1 = TargetedEstimation.Outputs(hdf5=TargetedEstimation.HDF5Output(filename="tmle_output_1.hdf5"))
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
    tmle_output_2 = TargetedEstimation.Outputs(hdf5=TargetedEstimation.HDF5Output(filename="tmle_output_2.hdf5"))
    config_2 = causal_and_joint_estimands_config()
    configfile_2 = joinpath(tmpdir, "configuration_2.json")
    TMLE.write_json(configfile_2, config_2)
    tmle(datafile; 
        estimands=configfile_2, 
        estimators=estimatorfile, 
        outputs=tmle_output_2
    )

    # Using the main entry point
    main([
        "merge", 
        "tmle_output", 
        "--json-output", "summary.json", 
        "--jls-output", "summary.jls",
        "--hdf5-output", "summary.hdf5"
    ])

    # Test correctness
    hdf5file_1 = jldopen("tmle_output_1.hdf5")
    hdf5file_2 = jldopen("tmle_output_2.hdf5")
    inputs = vcat(hdf5file_1["Batch_1"], hdf5file_1["Batch_2"], hdf5file_2["Batch_1"])

    json_outputs = TMLE.read_json("summary.json", use_mmap=false)
    jls_outputs = []
    open("summary.jls") do io
        while !eof(io)
            push!(jls_outputs, deserialize(io))
        end
    end
    hdf5_output = jldopen("summary.hdf5")
    hdf5_outputs = vcat((hdf5_output[key] for key in keys(hdf5_output))...)

    @test length(inputs) == 9
    for (input, jls_output, hdf5_out, json_output) in zip(inputs, jls_outputs, hdf5_outputs, json_outputs)
        @test input.OSE.estimand == jls_output.OSE.estimand == hdf5_out.OSE.estimand == json_output[:OSE].estimand
    end

    close(hdf5file_1)
    close(hdf5file_2)
    close(hdf5_output)

    # cleanup
    rm("tmle_output_1.hdf5")
    rm("tmle_output_2.hdf5")
    rm("summary.hdf5")
    rm("summary.jls")
    rm("summary.json")
    rm(datafile)
end


end

true