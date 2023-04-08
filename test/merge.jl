module TestMergeCSVFiles

using TargetedEstimation
using Test
using CSV
using DataFrames

@testset "Test merge_csv_files, no sieve file" begin
    parsed_args = Dict(
        "tmle-prefix" => joinpath("data", "merge", "tmle"),
        "sieve-prefix" => nothing,
        "out" => "output.csv"
    )
    merge_csv_files(parsed_args)
    output = CSV.read(parsed_args["out"], DataFrame)
    @test names(output) == [
        "PARAMETER_TYPE", "TREATMENTS", "CASE",
        "CONTROL", "TARGET", "CONFOUNDERS",
        "COVARIATES", "INITIAL_ESTIMATE", 
        "TMLE_ESTIMATE", "TMLE_STD", "TMLE_PVALUE", "TMLE_LWB", "TMLE_UPB",
        "ONESTEP_ESTIMATE", "ONESTEP_STD", "ONESTEP_PVALUE", "ONESTEP_LWB", "ONESTEP_UPB", 
        "LOG"
    ]
    @test size(output, 1) == 8
    @test output.PARAMETER_TYPE == [
        "IATE", "IATE", "ATE",
        "IATE", "IATE", "ATE",
        "ATE", "CM"
    ]
    rm(parsed_args["out"])
end

@testset "Test merge_csv_files, sieve file" begin
    # df = CSV.read("data/merge/sieve_output_1.csv", DataFrame)
    # rename!(df, [:ESTIMATE => :TMLE_ESTIMATE, :STD => :TMLE_STD, :PVALUE => :TMLE_PVALUE, :LWB => :TMLE_LWB, :UPB => :TMLE_UPB])
    # ext = DataFrame(
    #     ONESTEP_ESTIMATE = rand(2),
    #     ONESTEP_STD = rand(2),
    #     ONESTEP_PVALUE = rand(2),
    #     ONESTEP_LWB = rand(2),
    #     ONESTEP_UPB = rand(2),
    #     LOG = df.LOG,
    #     )
    # new_df = hcat(select!(df, Not(:LOG)), ext)
    # CSV.write("data/merge/tmle_output_2.csv", new_df)
    sieve_colnames = [
        "PARAMETER_TYPE", "TREATMENTS", "CASE",
        "CONTROL", "TARGET", "CONFOUNDERS",
        "COVARIATES", "INITIAL_ESTIMATE", 
        "TMLE_ESTIMATE", "TMLE_STD", "TMLE_PVALUE", "TMLE_LWB", "TMLE_UPB", 
        "ONESTEP_ESTIMATE", "ONESTEP_STD", "ONESTEP_PVALUE", "ONESTEP_LWB", "ONESTEP_UPB",
        "LOG", "SIEVE_STD", "SIEVE_PVALUE", "SIEVE_LWB", "SIEVE_UPB"
    ]
    parsed_args = Dict(
        "tmle-prefix" => joinpath("data", "merge", "tmle"),
        "sieve-prefix" => joinpath("data", "merge", "sieve"),
        "out" => "output.csv"
    )
    merge_csv_files(parsed_args)
    output = CSV.read(parsed_args["out"], DataFrame)
    @test names(output) == sieve_colnames
    @test size(output, 1) == 8
    @test output.SIEVE_STD isa Vector{Float64}
    @test output.PARAMETER_TYPE == [
        "IATE", "IATE", "ATE",
        "IATE", "IATE", "ATE",
        "ATE", "CM"
    ]

    parsed_args = Dict(
        "tmle-prefix" => joinpath("data", "merge", "tmle"),
        "sieve-prefix" => joinpath("data", "merge", "sieve_output_2"),
        "out" => "output.csv"
    )
    merge_csv_files(parsed_args)
    output = CSV.read(parsed_args["out"], DataFrame)
    @test names(output) == sieve_colnames
    @test size(output, 1) == 8
    @test all(x===missing for x in output.SIEVE_STD[3:end])

    rm(parsed_args["out"])
end

end

true