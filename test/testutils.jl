using TMLE

function statistical_estimands_only_config()
    configuration = Configuration(
        estimands=[
            IATE(
                outcome = Symbol("CONTINUOUS, OUTCOME"), 
                treatment_values = (
                    T1 = (case = true, control = false), 
                    T2 = (case = true, control = false)), 
                treatment_confounders = (T1 = (:W1, :W2), T2 = (:W1, :W2)), 
                outcome_extra_covariates = (:C1,)
            ),
            ATE(
                outcome = Symbol("CONTINUOUS, OUTCOME"), 
                treatment_values = (T1 = (case = true, control = false),), 
                treatment_confounders = (T1 = (:W1, :W2),), 
                outcome_extra_covariates = ()
            ),
            IATE(
                outcome = Symbol("CONTINUOUS, OUTCOME"), 
                treatment_values = (
                    T1 = (case = true, control = false), 
                    T2 = (case = false, control = true)
                ), 
                treatment_confounders = (T1 = (:W1, :W2), T2 = (:W1, :W2)), 
                outcome_extra_covariates = ()
            ),
            IATE(
                outcome = Symbol("BINARY/OUTCOME"), 
                treatment_values = (
                    T1 = (case = true, control = false), 
                    T2 = (case = false, control = true)
                ), 
                treatment_confounders = (T1 = (:W1, :W2), T2 = (:W1, :W2)), 
                outcome_extra_covariates = (:C1,)
            ),
            IATE(
                outcome = Symbol("BINARY/OUTCOME"), 
                treatment_values = (
                    T1 = (case = true, control = false), 
                    T2 = (case = true, control = false)), 
                treatment_confounders = (T1 = (:W1, :W2), T2 = (:W1, :W2)), 
                outcome_extra_covariates = (:C1,)
            ),
            CM(
                outcome = Symbol("CONTINUOUS, OUTCOME"), 
                treatment_values = (
                    T1 = true, 
                    T2 = false), 
                treatment_confounders = (T1 = (:W1, :W2), T2 = (:W1, :W2)),
                outcome_extra_covariates = (:C1,)
            )
        ]
    )
    return configuration
end

function causal_and_composed_estimands_config()
    ATE₁ = ATE(
        outcome = Symbol("CONTINUOUS, OUTCOME"), 
        treatment_values = (T1 = (case = true, control = false),), 
    )
    ATE₂ = ATE(
        outcome = Symbol("CONTINUOUS, OUTCOME"), 
        treatment_values = (T1 = (case = false, control = true),), 
    )
    diff = ComposedEstimand(-, (ATE₁, ATE₂))
    scm = StaticSCM(
        outcomes = ["CONTINUOUS, OUTCOME"],
        treatments = ["T1"],
        confounders = [:W1, :W2]
    )
    configuration = Configuration(
        estimands = [ATE₁, ATE₂, diff],
        scm       = scm
    )
    return configuration
end
