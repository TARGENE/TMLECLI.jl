module TestsStackBuilding

using Test
using TargetedEstimation
using MLJ
using MLJGLMInterface
using MLJLinearModels
using EvoTrees

@testset "Test tmle_spec_from_yaml: Only Stacks" begin
    tmle_spec = TargetedEstimation.load_tmle_spec(joinpath("config", "tmle_config.jl"))

    @test tmle_spec.threshold == 0.001
    # Test binary target TMLE's Qstack
    Q_binary = tmle_spec.Q_binary
    @test Q_binary.cache == false
    ## Checking Qstack.metalearner
    @test Q_binary.metalearner isa LogisticClassifier
    @test Q_binary.metalearner.fit_intercept == false
    ## Checking Qstack.resampling
    @test Q_binary.resampling isa StratifiedCV
    @test Q_binary.resampling.nfolds == 2
    ## Checking Qstack EvoTree models
    @test Q_binary.gridsearch_evo.tuning.goal == 5
    @test Q_binary.gridsearch_evo.cache == false
    @test Q_binary.gridsearch_evo.model.nrounds == 10
    @test Q_binary.gridsearch_evo.resampling isa CV
    ranges = Q_binary.gridsearch_evo.range
    @test ranges[2].lower == 1e-5
    @test ranges[2].upper == 10
    @test ranges[2].scale == :log
    @test ranges[1].lower == 3
    @test ranges[1].upper == 5
    @test ranges[1].scale == :linear
    ## Checking Qstack  Interaction Logistic models
    @test Q_binary.interaction_glmnet isa MLJ.ProbabilisticPipeline
    @test Q_binary.interaction_glmnet.interaction_transformer.order == 2
    ## Checking Qstack HAL model
    @test Q_binary.hal.lambda == 10
    @test Q_binary.hal.smoothness_orders == 1
    @test Q_binary.hal.cv_select == false
    @test Q_binary.hal.num_knots == [10, 5]

    # Test continuous target TMLE's Qstack
    Q_continuous = tmle_spec.Q_continuous
    ## Checking Qstack.metalearner
    @test Q_continuous.metalearner isa MLJLinearModels.LinearRegressor
    @test Q_continuous.metalearner.fit_intercept == false

    ## Checking Qstack.resampling
    @test Q_continuous.resampling isa CV
    @test Q_continuous.resampling.nfolds == 2
    ## Checking Qstack EvoTree models
    @test Q_continuous.evo_10.nrounds == 10
    @test Q_continuous.evo_20.nrounds == 20
    ## Checking Qstack Interaction Linear model
    @test Q_continuous.interaction_glmnet isa MLJ.DeterministicPipeline
    @test Q_continuous.interaction_glmnet.interaction_transformer.order == 3
    @test Q_continuous.interaction_glmnet.interaction_transformer.primary_variables == []
    @test Q_continuous.interaction_glmnet.interaction_transformer.primary_variables_patterns == [r"^rs[0-9]+"]
    ## Checking Qstack HAL model
    @test Q_continuous.hal.lambda == 10
    @test Q_continuous.hal.smoothness_orders == 1
    @test Q_continuous.hal.cv_select == false
    @test Q_continuous.hal.num_knots == [10, 5]
    
    # TMLE G Stack
    G = tmle_spec.G
    ## Checking Gstack.metalearner
    @test G.metalearner isa LogisticClassifier
    @test G.metalearner.fit_intercept == false
    ## Checking Gstack.resampling
    @test G.resampling isa StratifiedCV
    @test G.resampling.nfolds == 2
    ## Checking Gstack models
    @test G.interaction_glmnet.interaction_transformer.order == 2
    @test G.interaction_glmnet.interaction_transformer.primary_variables == [:T1, :T2]
    @test G.interaction_glmnet.interaction_transformer.primary_variables_patterns == [r"C"]
    @test G.evo.nrounds == 10

    @test tmle_spec.cache == false
end

@testset "Test tmle_spec_from_yaml: Simple models and GridSearch" begin
    tmle_spec = TargetedEstimation.load_tmle_spec(joinpath("config", "tmle_config_2.jl"))
    @test tmle_spec.G.cache == true
    @test tmle_spec.G.measure isa LogLoss
    @test tmle_spec.G.tuning.goal == 5
    @test tmle_spec.G.model.nrounds == 10
    lambda_range = tmle_spec.G.range[2]
    @test lambda_range.lower == 1e-5
    @test lambda_range.upper == 10
    @test lambda_range.scale == :log
    depth_range = tmle_spec.G.range[1]
    @test depth_range.lower == 3
    @test depth_range.upper == 5
    @test depth_range.scale == :linear

    @test tmle_spec.Q_binary isa MLJ.ProbabilisticPipeline
    @test tmle_spec.threshold == 1e-8

    @test tmle_spec.Q_continuous.cache == true
    @test tmle_spec.Q_continuous.interaction_glmnet.cache == true

    @test tmle_spec.cache == true
end

end;

true

