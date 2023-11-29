module TestOutputs

using TargetedEstimation
using Test
using JSON

TESTDIR = joinpath(pkgdir(TargetedEstimation), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test initialize" begin
    outputs = TargetedEstimation.Outputs(
        json = TargetedEstimation.JSONOutput(filename="output.json"),
        jls = TargetedEstimation.JLSOutput(filename="output.jls"),
        hdf5 = TargetedEstimation.HDF5Output(filename="output.hdf5"),
    )

    TargetedEstimation.initialize(outputs)

    @test isfile(outputs.json.filename)
    @test_throws TargetedEstimation.FileExistsError(outputs.json.filename) TargetedEstimation.initialize(outputs)
    rm(outputs.json.filename)

    touch(outputs.jls.filename)
    @test_throws TargetedEstimation.FileExistsError(outputs.jls.filename) TargetedEstimation.initialize(outputs)
    rm(outputs.jls.filename)
    rm(outputs.json.filename)

    touch(outputs.hdf5.filename)
    @test_throws TargetedEstimation.FileExistsError(outputs.hdf5.filename) TargetedEstimation.initialize(outputs)
    rm(outputs.hdf5.filename)
    rm(outputs.json.filename)
end

@testset "Test JSON update_file" begin
    results = []
    for Ψ in statistical_estimands_only_config().estimands
        push!(results, (
            TMLE=TMLE.TMLEstimate(Ψ, rand(), rand(), 10, Float64[]),
            OSE=TMLE.OSEstimate(Ψ, rand(), rand(), 10, Float64[])
            ))
    end
    tmpdir = mktempdir(cleanup=true)
    jsonoutput = TargetedEstimation.JSONOutput(filename=joinpath(tmpdir, "output_test.json"))
    TargetedEstimation.initialize_json(jsonoutput.filename)
    TargetedEstimation.update_file(jsonoutput, results[1:3])
    TargetedEstimation.update_file(jsonoutput, results[4:end]; finalize=true)
    loaded_results = TMLE.read_json(jsonoutput.filename)
    @test size(loaded_results) == size(results)
    for (result, loaded_result) in zip(results, loaded_results)
        @test result.TMLE.estimate == loaded_result[:TMLE].estimate
        @test result.TMLE.std == loaded_result[:TMLE].std
        @test result.OSE.estimate == loaded_result[:OSE].estimate
        @test result.OSE.std == loaded_result[:OSE].std
    end
end

end

true