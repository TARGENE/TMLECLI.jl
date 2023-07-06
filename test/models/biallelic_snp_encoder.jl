module TestBiAllelelicSNPEncoder

using Test
using TargetedEstimation
using CategoricalArrays
using MLJBase

@testset "Test BiAllelelicSNPEncoder" begin
    X = (
        rs1234 = categorical(["AC", "CC", "CC", missing]),
        rs4567 = categorical(["GG", "GC", "CC", "CC"]),
        othercol = [1, 2, 3, 4]
    )
    model = BiAllelicSNPEncoder(patterns=[r"^rs12", "^rs45"])
    @test model.patterns == [r"^rs12", r"^rs45"]
    mach = machine(model, X)
    fit!(mach, verbosity=0)
    fitresult = fitted_params(mach).fitresult
    @test fitresult == Dict(:rs1234 => 'A', :rs4567 => 'C')
    Xt = transform(mach)
    @test Xt.rs1234[1:3] == [1, 0, 0]
    @test Xt.rs1234[4] === missing
    @test Xt.rs4567 == [0, 1, 2, 2]
    @test Xt.othercol == [1, 2, 3, 4]

    X = (
        rs1234 = categorical(["AC", "CC", "CCC", missing]),
        othercol = [1, 2, 3, 4]
    )
    mach = machine(BiAllelicSNPEncoder(patterns=[r"^rs"]), X)
    @test_throws TargetedEstimation.NonBiAllelicGenotypeError(:rs1234, "CCC") fit!(mach, verbosity=0)

    X = (
        rs1234 = categorical(["AC", "CC", "CCT", missing]),
        othercol = [1, 2, 3, 4]
    )
    mach = machine(BiAllelicSNPEncoder(patterns=[r"^rs"]), X)
    @test_throws TargetedEstimation.NonBiallelicSNPError(:rs1234) fit!(mach, verbosity=0)

    X = (
        rs1234 = ["AC", "CC", "CCT", missing],
        othercol = [1, 2, 3, 4]
    )
    mach = machine(BiAllelicSNPEncoder(patterns=[r"^rs"]), X)
    @test_throws TargetedEstimation.NonCategoricalVectorError(:rs1234) fit!(mach, verbosity=0)


end

end

