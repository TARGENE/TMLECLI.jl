module TestResampling

using Test
using TargetedEstimation
using CategoricalArrays
using MLJBase
using StableRNGs

#####################################################################
#####                      AdativeResampling                    #####
#####################################################################

@testset "Test AdativeResampling's methods" begin
    @test TargetedEstimation.base_resampling(AdaptiveCV()) == CV
    @test TargetedEstimation.base_resampling(AdaptiveStratifiedCV()) == StratifiedCV
end

@testset "Test AdaptiveCV" begin
    # Continuous target
    cv = AdaptiveCV()
    n_sample_to_nfolds = ((10, 10), (200, 20), (1000, 10), (5000, 5), (20000, 3))
    for (n, expected_nfolds) in n_sample_to_nfolds
        y = rand(n)
        ttp = MLJBase.train_test_pairs(cv, 1:n, y)
        @test length(ttp) == expected_nfolds
        @test ttp == MLJBase.train_test_pairs(CV(nfolds=expected_nfolds), 1:n, y)
    end
    # Categorical target
    cv = AdaptiveStratifiedCV()
    y = categorical(["a", "a", "a", "b", "b", "c", "c"])
    @test TargetedEstimation.countuniques(y) == [3, 2, 2]
    ## neff < 30 => nfolds = 5*neff = 7
    ttp = MLJBase.train_test_pairs(cv, 1:7, y)
    @test length(ttp) == 7
    @test ttp == MLJBase.train_test_pairs(StratifiedCV(nfolds=7), 1:7, y)
    
    ## neff = 2500 => 10
    n = 50_500
    y = categorical(vcat(repeat([true], 50_000), repeat([false], 500)))
    @test TargetedEstimation.countuniques(y) == [50_000, 500]
    ttp = MLJBase.train_test_pairs(cv, 1:n, y)
    @test length(ttp)== 10
    @test ttp == MLJBase.train_test_pairs(StratifiedCV(nfolds=10), 1:n, y)
end

#####################################################################
#####                 JointStratifiedCV                         #####
#####################################################################


@testset "Test JointStratifiedCV's methods" begin
    X = (
        X1 = [0, 0, 1, 0, 1, 0, missing],
        X2 = categorical([1, 2, 4, 4, 2, 2, missing]),
        X3 = [1.2, 2.3, 4.5, -4.5, 0.0, -3., 8.]
    )
    y = categorical([1, 0, 1, 1, 0, 1, 1])

    stratification_col = TargetedEstimation.initialize_aggregate(y)
    @test all(stratification_col .== "")

    # No pattern specified, all X finite variables are considered, i.e. X1 and X2
    TargetedEstimation.aggregate_features!(stratification_col, nothing, X)
    @test stratification_col == ["_0_1", "_0_2", "_1_4", "_0_4", "_1_2", "_0_2", "_missing_missing"]
    # y is finite and is considered
    TargetedEstimation.aggretate_finite_col!(stratification_col, y)
    @test stratification_col == ["_0_1_1", "_0_2_0", "_1_4_1", "_0_4_1", "_1_2_0", "_0_2_1", "_missing_missing_1"]

    y = [1., 1.1, 2., 5., -4., -4., 3.2]
    stratification_col = TargetedEstimation.initialize_aggregate(y)
    # Only X1 will be matched
    TargetedEstimation.aggregate_features!(stratification_col, [r"X1"], X)
    @test stratification_col == ["_0", "_0", "_1", "_0", "_1", "_0", "_missing"]
    # y is continuous and is not considered
    TargetedEstimation.aggretate_finite_col!(stratification_col, y)
    @test stratification_col == ["_0", "_0", "_1", "_0", "_1", "_0", "_missing"]
end

@testset "Test JointStratifiedCV" begin
    X = (
        X1 = [0, 0, 1, 0, 1, 0, missing],
        X2 = categorical([1, 2, 4, 4, 2, 2, missing]),
        X3 = [1.2, 2.3, 4.5, -4.5, 0.0, -3., 8.]
    )
    y = categorical([1, 0, 1, 1, 0, 1, 1])

    nfolds = 3
    resampling = JointStratifiedCV(resampling=StratifiedCV(nfolds=nfolds, rng=StableRNG(123)))
    ttp = MLJBase.train_test_pairs(resampling, 1:nrows(y), X, y)
    @test length(ttp) == nfolds
end

@testset "Test JointStratifiedCV with AdativeStratifiedCV" begin
    X = (
        X1 = [0, 0, 1, 0, 1, 0, missing],
        X2 = categorical([1, 2, 4, 4, 2, 2, missing]),
        X3 = [1.2, 2.3, 4.5, -4.5, 0.0, -3., 8.]
    )
    y = categorical([1, 0, 1, 1, 0, 1, 1])

    nfolds = 3
    resampling = JointStratifiedCV(resampling=AdaptiveStratifiedCV())
    ttp = MLJBase.train_test_pairs(resampling, 1:nrows(y), X, y)
    @test length(ttp) == 5
end

true

end