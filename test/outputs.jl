module TestOutputs

using TMLECLI
using Test
using JSON

TESTDIR = joinpath(pkgdir(TMLECLI), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test initialize" begin
    tmpdir = mktempdir()
    outputs = TMLECLI.Outputs(
        json = joinpath(tmpdir, "output.json"),
        jls = joinpath(tmpdir, "output.jls"),
        hdf5 = joinpath(tmpdir, "output.hdf5"),
    )

    TMLECLI.initialize(outputs)

    @test readlines(open(outputs.json)) == ["["]
    @test !isfile(outputs.jls)
    @test !isfile(outputs.hdf5)

    # Initialize removes existing files
    touch(outputs.jls)
    touch(outputs.hdf5)
    @test isfile(outputs.jls)
    @test isfile(outputs.hdf5)
    TMLECLI.initialize(outputs)
    @test readlines(open(outputs.json)) == ["["]
    @test !isfile(outputs.jls)
    @test !isfile(outputs.hdf5)
end

@testset "Test update_json" begin
    results = []
    for Ψ in statistical_estimands_only_config().estimands
        push!(results, (
            TMLE=TMLE.TMLEstimate(Ψ, rand(), rand(), 10, Float64[]),
            OSE=TMLE.OSEstimate(Ψ, rand(), rand(), 10, Float64[])
            ))
    end
    tmpdir = mktempdir()
    filename = joinpath(tmpdir, "output_test.json")
    TMLECLI.initialize_json(filename)
    TMLECLI.update_json(filename, results[1:3])
    TMLECLI.update_json(filename, results[4:end])
    TMLECLI.finalize_json(filename)
    loaded_results = TMLE.read_json(filename, use_mmap=false)
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