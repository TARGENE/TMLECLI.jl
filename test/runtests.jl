using TMLECLI
using Test

TESTDIR = joinpath(pkgdir(TMLECLI), "test")

@time begin
    @test include(joinpath(TESTDIR, "outputs.jl"))
    @test include(joinpath(TESTDIR, "cache_managers.jl"))
    @test include(joinpath(TESTDIR, "utils.jl"))
    @test include(joinpath(TESTDIR, "sieve_variance.jl"))
    @test include(joinpath(TESTDIR, "runner.jl"))
    @test include(joinpath(TESTDIR, "summary.jl"))
    @test include(joinpath(TESTDIR, "resampling.jl"))
    @test include(joinpath(TESTDIR, "models", "glmnet.jl"))
    @test include(joinpath(TESTDIR, "models", "adaptive_interaction_transformer.jl"))
    @test include(joinpath(TESTDIR, "models", "biallelic_snp_encoder.jl"))
    @test include(joinpath(TESTDIR, "models", "registry.jl"))
end