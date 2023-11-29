module TestMergeCSVFiles

using TargetedEstimation
using Test
using CSV
using DataFrames

@testset "Test merge_csv_files, no sieve file" begin
    make_summary("tmle_out")
end


end

true