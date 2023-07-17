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
        "LOG", "TRAIT_ADJUSTED_TMLE_PVALUE"
    ]
    @test size(output, 1) == 8

    for (pval, adjusted_pval) in zip(output.TMLE_PVALUE, output.TRAIT_ADJUSTED_TMLE_PVALUE)
        if pval === missing
            @test adjusted_pval === missing
        else
            @test pval <= adjusted_pval
        end
    end

    @test output.PARAMETER_TYPE == [
        "IATE", "IATE", "ATE",
        "IATE", "IATE", "ATE",
        "ATE", "CM"
    ]
    rm(parsed_args["out"])
end

@testset "Test merge_csv_files, sieve file" begin
    sieve_colnames = [
        "PARAMETER_TYPE", "TREATMENTS", "CASE",
        "CONTROL", "TARGET", "CONFOUNDERS",
        "COVARIATES", "INITIAL_ESTIMATE", 
        "TMLE_ESTIMATE", "TMLE_STD", "TMLE_PVALUE", "TMLE_LWB", "TMLE_UPB", 
        "ONESTEP_ESTIMATE", "ONESTEP_STD", "ONESTEP_PVALUE", "ONESTEP_LWB", "ONESTEP_UPB",
        "LOG", "SIEVE_STD", "SIEVE_PVALUE", "SIEVE_LWB", "SIEVE_UPB", "TRAIT_ADJUSTED_TMLE_PVALUE"
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

@testset "Test merge_csv_files, empty sieve file" begin
    parsed_args = Dict(
        "tmle-prefix" => joinpath("data", "merge", "tmle"),
        "sieve-prefix" => joinpath("data", "merge", "empty"),
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
        "LOG", "TRAIT_ADJUSTED_TMLE_PVALUE"
    ]
    @test size(output, 1) == 8
    @test output.PARAMETER_TYPE == [
        "IATE", "IATE", "ATE",
        "IATE", "IATE", "ATE",
        "ATE", "CM"
    ]

    rm(parsed_args["out"])
end


end

true