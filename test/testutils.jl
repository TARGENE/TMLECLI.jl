using TMLE
using StableRNGs
using DataFrames
using Distributions
using LogExpFunctions
using CSV
using Arrow
using CategoricalArrays

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
                outcome = Symbol("COUNT_OUTCOME"), 
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

function causal_and_joint_estimands_config()
    ATE₁ = ATE(
        outcome = Symbol("CONTINUOUS, OUTCOME"), 
        treatment_values = (T1 = (case = true, control = false),), 
    )
    ATE₂ = ATE(
        outcome = Symbol("CONTINUOUS, OUTCOME"), 
        treatment_values = (T1 = (case = false, control = true),), 
    )
    joint = JointEstimand(ATE₁, ATE₂)
    scm = StaticSCM(
        outcomes = ["CONTINUOUS, OUTCOME"],
        treatments = ["T1"],
        confounders = [:W1, :W2]
    )
    configuration = Configuration(
        estimands = [ATE₁, ATE₂, joint],
        scm       = scm
    )
    return configuration
end

"""
CONTINUOUS_OUTCOME: 
- IATE(0->1, 0->1) = E[W₂] = 0.5
- ATE(0->1, 0->1)  = -4 E[C₁] + 1 + E[W₂] = -2 + 1 + 0.5 = -0.5

BINARY_OUTCOME:
- IATE(0->1, 0->1) =
- ATE(0->1, 0->1)  = 

"""
function build_dataset(;n=1000, format="csv")
    rng = StableRNG(123)
    # Confounders
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< logistic.(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< logistic.(-3W₁ - 1.5W₂)
    # target | Confounders, Covariates, Treatments
    μ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .+ T₁ + T₂.*W₂.*T₁
    y₁ = μ .+ rand(rng, Normal(0, 0.01), n)
    y₂ = rand(rng, Uniform(), n) .< logistic.(μ)
    # Add some missingness
    y₂ = vcat(missing, y₂[2:end])

    dataset = DataFrame(
        SAMPLE_ID = 1:n,
        T1 = T₁,
        T2 = T₂,
        W1 = W₁, 
        W2 = W₂,
        C1 = C₁,
    )
    # Comma in name
    dataset[!, "CONTINUOUS, OUTCOME"] = y₁
    # Slash in name
    dataset[!, "BINARY/OUTCOME"] = y₂
    dataset[!, "COUNT_OUTCOME"] = rand(rng, [1, 2, 3, 4], n)

    return dataset
end

function write_dataset(;n=1000, format="csv")
    dataset = build_dataset(;n=1000)
    format == "csv" ? CSV.write("data.csv", dataset) : Arrow.write("data.arrow", dataset)
end