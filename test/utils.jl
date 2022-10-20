module TestUtils

using Test
using TargetedEstimation
using MLJBase
using TMLE


@testset "Test AdaptiveCV" begin
    # Continuous target
    cv = TargetedEstimation.AdaptiveCV(CV())
    n_sample_to_nfolds = ((10, 10), (200, 20), (1000, 10), (5000, 5), (20000, 3))
    for (n, expected_nfolds) in n_sample_to_nfolds
        y = rand(n)
        ttp = MLJBase.train_test_pairs(cv, 1:n, y)
        @test length(ttp) == expected_nfolds
        @test ttp == MLJBase.train_test_pairs(CV(nfolds=expected_nfolds), 1:n, y)
    end
    # Categorical target
    cv = TargetedEstimation.AdaptiveCV(StratifiedCV(nfolds=2))
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

@testset "Test CSV writing" begin
    Ψ = TMLE.IATE(
        target=:Y,
        treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
        confounders=[:W₁, :W₂]
    )
    @test TargetedEstimation.covariates_string(Ψ) === missing
    @test TargetedEstimation.param_string(Ψ) == "IATE"
    @test TargetedEstimation.case_string(Ψ) == "1_&_AC"
    @test TargetedEstimation.control_string(Ψ) == "0_&_CC"
    @test TargetedEstimation.treatment_string(Ψ) == "T₁_&_T₂"
    @test TargetedEstimation.confounders_string(Ψ) == "W₁_&_W₂"

    Ψ = TMLE.CM(
        target=:Y,
        treatment=(T₁=1, T₂="AC"),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )

    @test TargetedEstimation.covariates_string(Ψ) === "C₁"
    @test TargetedEstimation.param_string(Ψ) == "CM"
    @test TargetedEstimation.case_string(Ψ) == "1_&_AC"
    @test TargetedEstimation.control_string(Ψ) === missing
    @test TargetedEstimation.treatment_string(Ψ) == "T₁_&_T₂"
    @test TargetedEstimation.confounders_string(Ψ) == "W₁_&_W₂"

end

end;

true