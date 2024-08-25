module TestRegistry

using TMLECLI
using Test
using TMLE
using MLJBase
using MLJXGBoostInterface
using MLJLinearModels

@testset "Test estimators_from_string: no models provided" begin
    # Default configuration results in GLMNets with interactions of order 2
    estimators = TMLECLI.estimators_from_string(config_string="wtmle-ose", treatment_variables=Set([:T1, :T2]))
    ## Check estimators
    @test estimators.WTMLE_GLMNET_GLMNET isa TMLEE
    @test estimators.WTMLE_GLMNET_GLMNET.weighted === true
    @test estimators.WTMLE_GLMNET_GLMNET.resampling === nothing

    @test estimators.OSE_GLMNET_GLMNET isa OSE
    @test estimators.OSE_GLMNET_GLMNET.resampling === nothing
    ## Check models
    expected_resampling = JointStratifiedCV(
        patterns = Regex[r"^T2$", r"^T1$"],
        resampling=StratifiedCV(nfolds=3)
        )
    for estimator in estimators
        Qbinary = estimator.models[:Q_binary_default].probabilistic_pipeline
        Qcontinuous = estimator.models[:Q_continuous_default].deterministic_pipeline
        G = estimator.models[:G_default]
        
        @test Qbinary.glm_net_classifier isa GLMNetClassifier
        @test Qbinary.glm_net_classifier.resampling == expected_resampling
        @test Qbinary.restricted_interaction_transformer.primary_variables_patterns == Regex[r"^T2$", r"^T1$"]
        
        @test Qcontinuous.glm_net_regressor isa GLMNetRegressor
        @test Qcontinuous.glm_net_regressor.resampling == expected_resampling
        @test Qcontinuous.restricted_interaction_transformer.primary_variables_patterns == Regex[r"^T2$", r"^T1$"]
        
        @test G.glm_net_classifier isa GLMNetClassifier
    end
end

@testset "Test estimators_from_string: 1 model provided" begin
    # 1 model is provided and used for all nuisance functions
    estimators = TMLECLI.estimators_from_string(config_string="cvtmle-cvose--tunedxgboost", treatment_variables=[])
    expected_resampling = JointStratifiedCV(
        patterns = Regex[],
        resampling=StratifiedCV(nfolds=3)
        )
    ## Check estimators
    @test estimators.CVTMLE_TUNEDXGBOOST_TUNEDXGBOOST isa TMLEE
    @test estimators.CVTMLE_TUNEDXGBOOST_TUNEDXGBOOST.weighted === false
    @test estimators.CVTMLE_TUNEDXGBOOST_TUNEDXGBOOST.resampling == expected_resampling

    @test estimators.CVOSE_TUNEDXGBOOST_TUNEDXGBOOST isa OSE
    @test estimators.CVOSE_TUNEDXGBOOST_TUNEDXGBOOST.resampling == expected_resampling
    ## Check models
    for estimator in estimators
        @test estimator.models[:Q_binary_default].probabilistic_tuned_model.model isa XGBoostClassifier
        @test estimator.models[:Q_continuous_default].deterministic_tuned_model.model isa XGBoostRegressor
        @test estimator.models[:G_default].probabilistic_tuned_model.model isa XGBoostClassifier
    end
end

@testset "Test estimators_from_string: 2 models provided" begin
    # 2 model is provided for nuisance functions
    estimators = TMLECLI.estimators_from_string(config_string="tmle--sl--glm", treatment_variables=["Coco"])
    ## Check estimators
    @test estimators.TMLE_SL_GLM isa TMLEE
    @test estimators.TMLE_SL_GLM.weighted === false
    @test estimators.TMLE_SL_GLM.resampling === nothing
    ## Check models
    expected_resampling = JointStratifiedCV(
        patterns = Regex[r"^Coco$"],
        resampling=StratifiedCV(nfolds=3)
        )
    for estimator in estimators
        Qbinary = estimator.models[:Q_binary_default].probabilistic_stack
        Qcontinuous = estimator.models[:Q_continuous_default].deterministic_stack
        G = estimator.models[:G_default].logistic_classifier

        @test Qbinary.glmnet.restricted_interaction_transformer.primary_variables_patterns == Regex[r"^Coco$"]
        @test Qbinary.glmnet.glm_net_classifier.resampling == expected_resampling
        @test Qbinary.lr.restricted_interaction_transformer.primary_variables_patterns == Regex[r"^Coco$"]
        @test Qbinary.lr.logistic_classifier isa LogisticClassifier
        xgboost_hyperparams = map(1:12) do i
            xgboost = getproperty(Qbinary, Symbol(:xgboost_classifier_, i))
            xgboost.eta, xgboost.max_depth
        end
        @test allunique(xgboost_hyperparams)

        @test Qcontinuous.glmnet.restricted_interaction_transformer.primary_variables_patterns == Regex[r"^Coco$"]
        @test Qcontinuous.glmnet.glm_net_regressor.resampling == expected_resampling
        @test Qcontinuous.lr.restricted_interaction_transformer.primary_variables_patterns == Regex[r"^Coco$"]
        @test Qcontinuous.lr.linear_regressor isa LinearRegressor
        xgboost_hyperparams = map(1:12) do i
            xgboost = getproperty(Qcontinuous, Symbol(:xgboost_regressor_, i))
            xgboost.eta, xgboost.max_depth
        end
        @test allunique(xgboost_hyperparams)

        @test G isa LogisticClassifier
    end
end

end

true