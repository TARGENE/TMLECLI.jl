module TestsStackBuilding

using Test
using TargetedEstimation
using MLJ
using MLJGLMInterface
using MLJLinearModels
using EvoTrees

@testset "Test tmle_spec_from_yaml: Only Stacks" begin
    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(joinpath("config", "tmle_config.yaml"))

    @test tmle_spec.threshold == 0.001
    # Test binary target TMLE's Qstack
    Q_binary = tmle_spec.Q_binary
    @test Q_binary.cache == false
    @test Q_binary.measures == [log_loss]
    ## Checking Qstack.metalearner
    @test Q_binary.metalearner isa LogisticClassifier
    @test Q_binary.metalearner.fit_intercept == false
    ## Checking Qstack.resampling
    @test Q_binary.resampling isa StratifiedCV
    @test Q_binary.resampling.nfolds == 2
    ## Checking Qstack EvoTree models
    @test Q_binary.GridSearchEvoTreeClassifier_1.tuning.goal == 5
    @test Q_binary.GridSearchEvoTreeClassifier_1.cache == false
    @test Q_binary.GridSearchEvoTreeClassifier_1.model.nrounds == 10
    @test Q_binary.GridSearchEvoTreeClassifier_1.resampling isa CV
    ranges = Q_binary.GridSearchEvoTreeClassifier_1.range
    @test ranges[1].lower == 1e-5
    @test ranges[1].upper == 10
    @test ranges[1].scale == :log
    @test ranges[2].lower == 3
    @test ranges[2].upper == 5
    @test ranges[2].scale == :linear
    ## Checking Qstack  Interaction Logistic models
    @test Q_binary.InteractionGLMNetClassifier_1 isa MLJ.ProbabilisticPipeline
    @test Q_binary.InteractionGLMNetClassifier_1.interaction_transformer.order == 2
    ## Checking Qstack HAL model
    @test Q_binary.HALClassifier_1.lambda == 10
    @test Q_binary.HALClassifier_1.smoothness_orders == 1
    @test Q_binary.HALClassifier_1.cv_select == false
    @test Q_binary.HALClassifier_1.num_knots == [10, 5]

    # Test continuous target TMLE's Qstack
    Q_continuous = tmle_spec.Q_continuous
    @test Q_continuous.measures == [rmse]
    ## Checking Qstack.metalearner
    @test Q_continuous.metalearner isa MLJLinearModels.LinearRegressor
    @test Q_continuous.metalearner.fit_intercept == false

    ## Checking Qstack.resampling
    @test Q_continuous.resampling isa CV
    @test Q_continuous.resampling.nfolds == 2
    ## Checking Qstack EvoTree models
    @test Q_continuous.EvoTreeRegressor_1.nrounds == 10
    @test Q_continuous.EvoTreeRegressor_2.nrounds == 20
    ## Checking Qstack Interaction Linear model
    @test Q_continuous.InteractionGLMNetRegressor_1 isa MLJ.DeterministicPipeline
    @test Q_continuous.InteractionGLMNetRegressor_1.interaction_transformer.order == 3
    ## Checking Qstack HAL model
    @test Q_continuous.HALRegressor_1.lambda == 10
    @test Q_continuous.HALRegressor_1.smoothness_orders == 1
    @test Q_continuous.HALRegressor_1.cv_select == false
    @test Q_continuous.HALRegressor_1.num_knots == [10, 5]
    
    # TMLE G Stack
    G = tmle_spec.G
    @test G.measures == [log_loss]
    ## Checking Gstack.metalearner
    @test G.metalearner isa LogisticClassifier
    @test G.metalearner.fit_intercept == false
    ## Checking Gstack.resampling
    @test G.resampling isa StratifiedCV
    @test G.resampling.nfolds == 2
    ## Checking Gstack models
    @test G.InteractionGLMNetClassifier_1.interaction_transformer.order == 2
    @test G.EvoTreeClassifier_1.nrounds == 10

    @test tmle_spec.cache == false
end

@testset "Test tmle_spec_from_yaml: Simple models and GridSearch" begin
    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(joinpath("config", "tmle_config_2.yaml"))
    @test tmle_spec.G.cache == true
    @test tmle_spec.G.measure isa LogLoss
    @test tmle_spec.G.tuning.goal == 5
    @test tmle_spec.G.model.nrounds == 10
    lambda_range = tmle_spec.G.range[1]
    @test lambda_range.lower == 1e-5
    @test lambda_range.upper == 10
    @test lambda_range.scale == :log
    depth_range = tmle_spec.G.range[2]
    @test depth_range.lower == 3
    @test depth_range.upper == 5
    @test depth_range.scale == :linear

    @test tmle_spec.Q_binary == TargetedEstimation.InteractionGLMNetClassifier()
    @test tmle_spec.threshold == 1e-8

    @test tmle_spec.Q_continuous.cache == true
    @test tmle_spec.Q_continuous.InteractionGLMNetRegressor_1.cache == true

    @test tmle_spec.cache == true
end

end;

true

