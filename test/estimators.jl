module TestsStackBuilding

using Test
using TargetedEstimation
using MLJBase
using MLJGLMInterface
using MLJLinearModels
using MLJModels
using EvoTrees

@testset "Test tmle_spec_from_yaml: Only Stacks" begin
    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(joinpath("config", "tmle_config.yaml"))

    @test tmle_spec.threshold == 0.001
    # Test binary target TMLE's Qstack
    Q_binary = tmle_spec.Q_binary
    @test Q_binary.measures == [log_loss]
    ## Checking Qstack.metalearner
    @test Q_binary.metalearner isa LogisticClassifier
    @test Q_binary.metalearner.fit_intercept == false
    ## Checking Qstack.resampling
    @test Q_binary.resampling isa StratifiedCV
    @test Q_binary.resampling.nfolds == 2
    ## Checking Qstack EvoTree models
    @test Q_binary.EvoTreeClassifier_1.nrounds == 10
    ## Checking Qstack  Interaction Logistic models
    @test Q_binary.InteractionLMClassifier_1 isa TargetedEstimation.InteractionLMClassifier
    @test Q_binary.InteractionLMClassifier_1.interaction_transformer.column_pattern == r"^RS_"
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
    @test Q_continuous.InteractionLMRegressor_1.interaction_transformer.column_pattern == r"^RS_"
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
    @test G.LogisticClassifier_1.lambda == 1.0
    @test G.EvoTreeClassifier_1.nrounds == 10
end

@testset "Test tmle_spec_from_yaml: Simple models" begin
    tmle_spec = TargetedEstimation.tmle_spec_from_yaml(joinpath("config", "tmle_config_2.yaml"))
    @test tmle_spec.G == EvoTreeClassifier(nrounds=10)
    @test tmle_spec.Q_binary == LogisticClassifier(lambda=10)
    @test tmle_spec.threshold == 1e-8
end

end;

true

