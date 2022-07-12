module TestsStackBuilding

using Test
using TargetedEstimation
using MLJBase
using TOML
using TMLE
using MLJGLMInterface
using MLJLinearModels
using MLJModels
using EvoTrees

include("testutils.jl")

@testset "Categorical target TMLE built from configuration file" begin
    queries = TargetedEstimation.parse_queries(iate_queryfile)
    tmle_bin = TargetedEstimation.tmle_from_yaml(tmle_configfile, queries, Bool)

    @test tmle_bin.threshold == 0.001
    # Test binary target TMLE's Qstack
    @test tmle_bin.Q̅.measures == [log_loss]
    @test tmle_bin.F isa LinearBinaryClassifier
    ## Checking Qstack.metalearner
    @test tmle_bin.Q̅.metalearner isa LogisticClassifier
    @test tmle_bin.Q̅.metalearner.fit_intercept == false
    ## Checking Qstack.resampling
    @test tmle_bin.Q̅.resampling isa StratifiedCV
    @test tmle_bin.Q̅.resampling.nfolds == 2
    ## Checking Qstack EvoTree models
    @test tmle_bin.Q̅.EvoTreeClassifier_1.nrounds == 10
    ## Checking Qstack  Interaction Logistic models
    @test tmle_bin.Q̅.InteractionLMClassifier_1 isa TargetedEstimation.InteractionLMClassifier
    @test tmle_bin.Q̅.InteractionLMClassifier_1.interaction_transformer.column_pattern == r"^RS_"
    ## Checking Qstack HAL model
    @test tmle_bin.Q̅.HALClassifier_1.lambda == 10
    @test tmle_bin.Q̅.HALClassifier_1.smoothness_orders == 1
    @test tmle_bin.Q̅.HALClassifier_1.cv_select == false
    @test tmle_bin.Q̅.HALClassifier_1.num_knots == [10, 5]

    # Test binary target TMLE Qstack
    tmle_cont = TargetedEstimation.tmle_from_yaml(tmle_configfile, queries, Real)
    @test tmle_cont.Q̅.measures == [rmse]
    @test tmle_cont.F isa MLJGLMInterface.LinearRegressor
    ## Checking Qstack.metalearner
    @test tmle_cont.Q̅.metalearner isa MLJLinearModels.LinearRegressor
    @test tmle_cont.Q̅.metalearner.fit_intercept == false

    ## Checking Qstack.resampling
    @test tmle_cont.Q̅.resampling isa CV
    @test tmle_cont.Q̅.resampling.nfolds == 2
    ## Checking Qstack EvoTree models
    @test tmle_cont.Q̅.EvoTreeRegressor_1.nrounds == 10
    @test tmle_cont.Q̅.EvoTreeRegressor_2.nrounds == 20
    ## Checking Qstack Interaction Linear model
    @test tmle_cont.Q̅.InteractionLMRegressor_1.interaction_transformer.column_pattern == r"^RS_"
    ## Checking Qstack HAL model
    @test tmle_cont.Q̅.HALRegressor_1.lambda == 10
    @test tmle_cont.Q̅.HALRegressor_1.smoothness_orders == 1
    @test tmle_cont.Q̅.HALRegressor_1.cv_select == false
    @test tmle_cont.Q̅.HALRegressor_1.num_knots == [10, 5]
    
    # Both TMLE have the same G Stack
    expected_queries = [
        Query(case=(RSID_10="AG", RSID_100="AG"), control=(RSID_10="GG", RSID_100="GG"), name="QUERY_1"),
        Query(case=(RSID_10="AG", RSID_100="AA"), control=(RSID_10="GG", RSID_100="GG"), name="QUERY_2"),
        Query(case=(RSID_10="AG", RSID_100="AA"), control=(RSID_10="TT", RSID_100="GG"), name="QUERY_3")
    ]
    for tmle in [tmle_bin, tmle_cont]
        @test tmle.G.model.measures == [log_loss]
        test_queries(tmle.queries, expected_queries)
        # Checking Gstack
        @test tmle.G isa FullCategoricalJoint 
        ## Checking Gstack.metalearner
        @test tmle.G.model.metalearner isa LogisticClassifier
        @test tmle.G.model.metalearner.fit_intercept == false
        ## Checking Gstack.resampling
        @test tmle.G.model.resampling isa StratifiedCV
        @test tmle.G.model.resampling.nfolds == 2
        ## Checking Gstack models
        @test tmle.G.model.LogisticClassifier_1.lambda == 1.0
        @test tmle.G.model.EvoTreeClassifier_1.nrounds == 10
    end
end

@testset "Test standard ATE TMLE build" begin
    queries = TargetedEstimation.parse_queries(ate_queryfile)
    tmle_bin = TargetedEstimation.tmle_from_yaml(tmle_configfile_2, queries, Bool)
    tmle_cont = TargetedEstimation.tmle_from_yaml(tmle_configfile_2, queries, Real)
    expected_queries = [
        Query(case=(RSID_10="AG",), control=(RSID_10="GG",), name="QUERY_1"),
        Query(case=(RSID_10="AA",), control=(RSID_10="GG",), name="QUERY_2"),
    ]
    for (type, tmle) in [("binary", tmle_bin), ("continuous", tmle_cont)]
        test_queries(tmle.queries, expected_queries)
        @test tmle.G isa EvoTreeClassifier
    end

    @test tmle_bin.Q̅ isa LogisticClassifier
    @test tmle_bin.Q̅.lambda == 10.

    @test tmle_cont.Q̅.EvoTreeRegressor_4 isa EvoTreeRegressor
    @test tmle_cont.Q̅.resampling isa TargetedEstimation.AdaptiveCV

end



end;

true

