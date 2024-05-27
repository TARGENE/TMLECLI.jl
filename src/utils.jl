#####################################################################
#####           Read TMLE Estimands Configuration                ####
#####################################################################

function convert_treatment_values(treatment_levels::NamedTuple{names, <:Tuple{Vararg{NamedTuple}}}, treatment_types) where names
    return [(
        case = convert(treatment_types[tn], treatment_levels[tn].case), 
        control = convert(treatment_types[tn], treatment_levels[tn].control)
    ) 
        for tn in names]
end

convert_treatment_values(treatment_levels::NamedTuple{names,}, treatment_types) where names = 
    [convert(treatment_types[tn], treatment_levels[tn]) for tn in names]

MissingSCMError() = ArgumentError(string("A Structural Causal Model should be provided in the configuration file in order to identify causal estimands."))

get_identification_method(method::Nothing) = BackdoorAdjustment()
get_identification_method(method) = method

function read_estimands_config(filename)
    if endswith(filename, ".json")
        TMLE.read_json(filename, use_mmap=false)
    elseif endswith(filename, ".yaml")
        TMLE.read_yaml(filename)
    elseif endswith(filename, ".jls")
        return deserialize(filename)
    else
        throw(ArgumentError(string("Can't read from ", extension, " file")))
    end
end

function fix_treatment_values!(treatment_types::AbstractDict, Ψ::JointEstimand, dataset)
    new_args = Tuple(fix_treatment_values!(treatment_types, arg, dataset) for arg in Ψ.args)
    return JointEstimand(new_args...)
end

wrapped_type(x) = x
wrapped_type(x::Type{<:CategoricalValue{T,}}) where T = T
wrapped_type(x::Type{Union{Missing, T}}) where T = wrapped_type(T)
"""
Uses the values found in the dataset to create a new estimand with adjusted values.
"""
function fix_treatment_values!(treatment_types::AbstractDict, Ψ, dataset)
    treatment_names = keys(Ψ.treatment_values)
    for tn in treatment_names
        haskey(treatment_types, tn) ? nothing : treatment_types[tn] = wrapped_type(eltype(dataset[!, tn]))
    end
    new_treatment = NamedTuple{treatment_names}(
        convert_treatment_values(Ψ.treatment_values, treatment_types)
    )
    return typeof(Ψ)(
        outcome = Ψ.outcome,
        treatment_values = new_treatment,
        treatment_confounders = Ψ.treatment_confounders,
        outcome_extra_covariates = Ψ.outcome_extra_covariates
    )
end

"""
    proofread_estimands(config, dataset)

Ensures that the treatment values in the config respect the treatment types in the dataset.
"""
function proofread_estimands(config, dataset)
    adjustment_method = get_identification_method(config.adjustment)
    estimands = Vector{TMLE.Estimand}(undef, length(config.estimands))
    treatment_types = Dict()
    for (index, Ψ) in enumerate(config.estimands)
        statisticalΨ = identify(Ψ, config.scm, method=adjustment_method)
        estimands[index] = fix_treatment_values!(treatment_types, statisticalΨ, dataset)
    end
    return estimands
end

"""
This explicitely requires that the following columns belong to the dataset:

- `T`: for the treatment variable
- `Y`: for the outcome variable
- `^W`: for the confounding variables

All ATE parameters are generated.
"""
function factorialATE(dataset)
    colnames = names(dataset)
    "T" ∈ colnames || throw(ArgumentError("No column 'T' found in the dataset for the treatment variable."))
    "Y" ∈ colnames || throw(ArgumentError("No column 'Y' found in the dataset for the outcome variable."))
    confounding_variables = Tuple(name for name in colnames if occursin(r"^W", name))
    length(confounding_variables) > 0 || throw(ArgumentError("Could not find any confounding variable (starting with 'W') in the dataset."))
    
    return [factorialEstimand(ATE, (:T,), :Y; dataset=dataset, confounders=confounding_variables)]
end

instantiate_config(file::AbstractString) = read_estimands_config(file)
instantiate_config(config) = config

function instantiate_estimands(estimands_pattern, dataset)
    estimands = if estimands_pattern == "factorialATE"
        factorialATE(dataset)
    else
        config = instantiate_config(estimands_pattern)
        proofread_estimands(config, dataset)
    end
    return estimands
end

#####################################################################
#####                 ADDITIONAL METHODS                         ####
#####################################################################

TMLE.emptyIC(nt::NamedTuple{names}, pval_threshold) where names =
    NamedTuple{names}([TMLE.emptyIC(result, pval_threshold) for result in nt])

"""
    instantiate_dataset(path::String)

Returns a DataFrame wrapper around a dataset, either in CSV format.
"""
instantiate_dataset(path::AbstractString) =
    endswith(path, ".csv") ? CSV.read(path, DataFrame, ntasks=1) : DataFrame(Arrow.Table(path))

instantiate_dataset(dataset) = dataset

isbinary(col, dataset) = Set(unique(skipmissing(dataset[!, col]))) == Set([0, 1])

make_categorical(x::CategoricalVector, ordered) = x
make_categorical(x, ordered) = categorical(x, ordered=ordered)

function make_categorical!(dataset, colname::Union{String, Symbol}; infer_ordered=false)
    ordered = false
    if infer_ordered
        ordered = eltype(dataset[!, colname]) <: Real
    end
    dataset[!, colname] = make_categorical(dataset[!, colname], ordered)
end

function make_categorical!(dataset, colnames; infer_ordered=false)
    for colname in colnames
        make_categorical!(dataset, colname; infer_ordered=infer_ordered)
    end
end

make_float(x) = float(x)

make_float!(dataset, colname::Union{String, Symbol}) = 
    dataset[!, colname] = make_float(dataset[!, colname])

function make_float!(dataset, colnames)
    for colname in colnames
        make_float!(dataset, colname)
    end
end

function coerce_types!(dataset, colnames; rules=:few_to_finite)
    infered_types = autotype(dataset[!, colnames], rules)
    coerce!(dataset, infered_types)
end

"""
Outcomes and Treatment variables must be dealt with differently until there are models dealing specificaly with Count data.

- Outcomes need to be binary, i.e. OrderedFactor{2} or Continuous
- Treatments need to be Categorical
- Other variables can be dealt with either way
"""
function coerce_types!(dataset, Ψ::TMLE.Estimand)
    all_outcomes = outcomes(Ψ)
    for outcome in all_outcomes
        if isbinary(outcome, dataset)
            coerce_types!(dataset, [outcome], rules=:few_to_finite)
        else
            coerce_types!(dataset, [outcome], rules=:discrete_to_continuous)
        end
    end
    other_variables = collect(setdiff(variables(Ψ), all_outcomes))
    coerce_types!(dataset, other_variables, rules=:few_to_finite)
end

outcomes(Ψ::TMLE.Estimand) = Set([Ψ.outcome])

outcomes(Ψ::TMLE.JointEstimand) = union((outcomes(arg) for arg in Ψ.args)...)

variables(Ψ::TMLE.JointEstimand) = union((variables(arg) for arg in Ψ.args)...)

variables(Ψ::TMLE.Estimand) = Set([
    Ψ.outcome,
    keys(Ψ.treatment_values)...,
    Ψ.outcome_extra_covariates..., 
    Iterators.flatten(values(Ψ.treatment_confounders))...
    ])

TMLE.to_dict(nt::NamedTuple{names, <:Tuple{Vararg{Union{TMLE.EICEstimate, FailedEstimate, TMLE.JointEstimate}}}}) where names = 
    Dict(key => TMLE.to_dict(val) for (key, val) ∈ zip(keys(nt), nt))