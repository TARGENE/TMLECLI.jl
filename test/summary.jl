module TestMergeCSVFiles

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
    build_dataset()
    datafile = "data.csv"
    estimatorfile = joinpath(CONFIGDIR, "ose_config.jl")
    tmpdir = mktempdir(cleanup=true)
    # First Run
    tmle_output_1 = TargetedEstimation.Outputs(hdf5=TargetedEstimation.HDF5Output(filename="tmle_output_1.hdf5"))
    config_1 = statistical_estimands_only_config()
    configfile_1 = joinpath(tmpdir, "configuration_1.json")
    TMLE.write_json(configfile_1, config_1)
    tmle(datafile, configfile_1, estimatorfile; outputs=tmle_output_1, chunksize=3)
    
    # Second Run
    tmle_output_2 = TargetedEstimation.Outputs(hdf5=TargetedEstimation.HDF5Output(filename="tmle_output_2.hdf5"))
    config_2 = causal_and_composed_estimands_config()
    configfile_2 = joinpath(tmpdir, "configuration_2.json")
    TMLE.write_json(configfile_2, config_2)
    tmle(datafile, configfile_2, estimatorfile; outputs=tmle_output_2)

    # Make summary files
    outputs = TargetedEstimation.Outputs(
        json=TargetedEstimation.JSONOutput(filename="summary.json"),
        hdf5=TargetedEstimation.HDF5Output(filename="summary.hdf5"),
        jls=TargetedEstimation.JLSOutput(filename="summary.jls")
    )
    make_summary("tmle_output", outputs=outputs)

    # Test correctness
    hdf5file_1 = jldopen("tmle_output_1.hdf5")
    hdf5file_2 = jldopen("tmle_output_2.hdf5")
    inputs = vcat(hdf5file_1["Batch_1"], hdf5file_1["Batch_2"], hdf5file_2["Batch_1"])

    json_outputs = TMLE.read_json(outputs.json.filename)
    jls_outputs = []
    open(outputs.jls.filename) do io
        while !eof(io)
            push!(jls_outputs, deserialize(io))
        end
    end
    hdf5_output = jldopen(outputs.hdf5.filename)
    hdf5_outputs = vcat((hdf5_output[key] for key in keys(hdf5_output))...)

    @test length(inputs) == 9
    for (input, jls_output, hdf5_out, json_output) in zip(inputs, jls_outputs, hdf5_outputs, json_outputs)
        @test input.OSE.estimand == jls_output.OSE.estimand == hdf5_out.OSE.estimand == json_output[:OSE].estimand
    end

    # cleanup
    rm("tmle_output_1.hdf5")
    rm("tmle_output_2.hdf5")
    rm(outputs.json.filename)
    rm(outputs.jls.filename)
    rm(outputs.hdf5.filename)
    rm(datafile)
end


end

true