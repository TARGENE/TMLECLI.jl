

#####################################################################
#####                       CSV OUTPUT                           ####
#####################################################################


empty_tmle_output(;size=0) = DataFrame(
    PARAMETER_TYPE=Vector{String}(undef, size), 
    TREATMENTS=Vector{String}(undef, size), 
    CASE=Vector{String}(undef, size), 
    CONTROL=Vector{Union{Missing, String}}(undef, size), 
    OUTCOME=Vector{String}(undef, size), 
    CONFOUNDERS=Vector{String}(undef, size), 
    COVARIATES=Vector{Union{Missing, String}}(undef, size), 
    INITIAL_ESTIMATE=Vector{Union{Missing, Float64}}(undef, size), 
    TMLE_ESTIMATE=Vector{Union{Missing, Float64}}(undef, size),
    TMLE_STD=Vector{Union{Missing, Float64}}(undef, size),
    TMLE_PVALUE=Vector{Union{Missing, Float64}}(undef, size),
    TMLE_LWB=Vector{Union{Missing, Float64}}(undef, size),
    TMLE_UPB=Vector{Union{Missing, Float64}}(undef, size),
    ONESTEP_ESTIMATE=Vector{Union{Missing, Float64}}(undef, size),
    ONESTEP_STD=Vector{Union{Missing, Float64}}(undef, size),
    ONESTEP_PVALUE=Vector{Union{Missing, Float64}}(undef, size),
    ONESTEP_LWB=Vector{Union{Missing, Float64}}(undef, size),
    ONESTEP_UPB=Vector{Union{Missing, Float64}}(undef, size),
    LOG=Vector{Union{Missing, String}}(undef, size)
)

covariates_string(Ψ; join_string="_&_") = 
    length(Ψ.outcome_extra_covariates) != 0 ? join(Ψ.outcome_extra_covariates, join_string) : missing

function param_string(param::T) where T <: TMLE.Estimand
    str = string(T)
    return startswith(str, "TMLE.") ? str[6:end] : str
end

case(nt::NamedTuple) = nt.case
case(x) = x
case_string(Ψ; join_string="_&_") = join((case(x) for x in values(Ψ.treatment)), join_string)

control_string(t::Tuple{Vararg{NamedTuple}}; join_string="_&_") = 
    join((val.control for val in t), join_string)

control_string(t; join_string="_&_") = missing

control_string(Ψ::TMLE.Estimand; join_string="_&_") = 
    control_string(values(Ψ.treatment); join_string=join_string)

treatment_string(Ψ; join_string="_&_") = join(keys(Ψ.treatment), join_string)
confounders_string(Ψ; join_string="_&_") = join(Ψ.confounders, join_string)


function statistics_from_estimator(estimator)
    Ψ̂ = TMLE.estimate(estimator)
    std = √(var(estimator))
    testresult = OneSampleTTest(estimator)
    pval = pvalue(testresult)
    l, u = confint(testresult)
    return (Ψ̂, std, pval, l, u)
end

function statistics_from_result(result::TMLE.Estimate)
    Ψ̂₀ = result.initial
    # TMLE stats
    tmle_stats = statistics_from_estimator(result.tmle) 
    # OneStep stats
    onestep_stats = statistics_from_estimator(result.onestep)
    return Ψ̂₀, tmle_stats, onestep_stats
end

statistics_from_result(result::FailedEstimation) = 
    missing, 
    (missing, missing, missing, missing, missing), 
    (missing, missing, missing, missing, missing)

function append_csv(filename, tmle_results, logs)
    data = empty_tmle_output(size=size(tmle_results, 1))
    for (i, (result, log)) in enumerate(zip(tmle_results, logs))
        Ψ = result.parameter
        param_type = param_string(Ψ)
        treatments = treatment_string(Ψ)
        case = case_string(Ψ)
        control = control_string(Ψ)
        confounders = confounders_string(Ψ)
        covariates = covariates_string(Ψ)
        Ψ̂₀, tmle_stats, onestep_stats = statistics_from_result(result)
        data[i, :] = (
            param_type, treatments, case, control, string(Ψ.target), confounders, covariates, 
            Ψ̂₀, tmle_stats..., onestep_stats..., log
        )
    end
    CSV.write(filename, data, append=true, header=!isfile(filename))
end


#####################################################################
#####                       JLD2 OUTPUT                          ####
#####################################################################

update_jld2_output(jld2_file::Nothing, partition, tmle_results, dataset; pval_threshold=0.05) = nothing

function update_jld2_output(jld2_file::String, partition, tmle_results, dataset; pval_threshold=0.05)
    if jld2_file !== nothing
        jldopen(jld2_file, "a+", compress=true) do io
        # Append only with results passing the threshold
            previous_variables = nothing
            sample_ids_idx = nothing

            for (partition_index, param_index) in enumerate(partition)
                r = tmle_results[partition_index]
                if (r isa TMLE.Estimate) && (pvalue(OneSampleZTest(r.tmle)) <= pval_threshold)
                    current_variables = variables(r.parameter)
                    if previous_variables != current_variables
                        sample_ids = TargetedEstimation.get_sample_ids(dataset, current_variables)
                        io["$param_index/sample_ids"] = sample_ids
                        sample_ids_idx = param_index
                    end
                    io["$param_index/result"] = r
                    io["$param_index/sample_ids_idx"] = sample_ids_idx

                    previous_variables = current_variables
                end
            end
        end
    end
end

#####################################################################
#####                    Read Estimands                         ####
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

maybe_identify(Ψ::TMLE.CausalCMCompositeEstimands, scm::SCM, method) = 
    identify(get_identification_method(method), Ψ, scm)

maybe_identify(Ψ::TMLE.CausalCMCompositeEstimands, scm::Nothing, method) = throw(MissingSCMError())

maybe_identify(Ψ, scm, method) = Ψ

"""
    read_estimands(param_file, dataset)

Reads estimands from file and ensures that the treatment values in the config file
respects the treatment types in the dataset.
"""
function proofread_estimands_from_yaml(filename, dataset)
    config = configuration_from_yaml(filename)
    estimands = Vector{TMLE.Estimand}(undef, length(config.estimands))
    treatment_types = Dict()
    for (index, Ψ) in enumerate(config.estimands)
        statisticalΨ = TargetedEstimation.maybe_identify(Ψ, config.scm, config.adjustment)
        treatment_names = keys(statisticalΨ.treatment_values)
        for tn in treatment_names
            haskey(treatment_types, tn) ? nothing : treatment_types[tn] = eltype(dataset[!, tn])
        end
        new_treatment = NamedTuple{treatment_names}(
            TargetedEstimation.convert_treatment_values(statisticalΨ.treatment_values, treatment_types)
        )
        estimands[index] = typeof(Ψ)(
            outcome = Ψ.outcome,
            treatment_values = new_treatment,
            treatment_confounders = statisticalΨ.treatment_confounders,
            outcome_extra_covariates = statisticalΨ.outcome_extra_covariates
        )
    end
    return estimands
end

#####################################################################
#####                 ADDITIONAL METHODS                         ####
#####################################################################

function get_sample_ids(data, variables)
    cols = [:SAMPLE_ID, variables.target, variables.treatments..., variables.confounders..., variables.covariates...]
    return dropmissing(data[!, cols]).SAMPLE_ID
end

"""
    instantiate_dataset(path::String)

Returns a DataFrame wrapper around a dataset, either in CSV format.
"""
instantiate_dataset(path::String) =
    endswith(path, ".csv") ? CSV.read(path, DataFrame, ntasks=1) : DataFrame(Arrow.Table(path))

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
        make_categorical!(dataset, colname;infer_ordered=infer_ordered)
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

function coerce_types!(dataset, Ψ)
    categorical_variables = Set(keys(Ψ.treatment_values))
    continuous_variables = Set(Iterators.flatten(values(Ψ.treatment_confounders)))
    union!(continuous_variables, Ψ.outcome_extra_covariates) 
    TMLE.is_binary(dataset, Ψ.outcome) ? 
        push!(categorical_variables, Ψ.outcome) : 
        push!(continuous_variables, Ψ.outcome)
    make_categorical!(dataset, categorical_variables, infer_ordered=true)
    make_float!(dataset, continuous_variables)
end

variables(Ψ::TMLE.Estimand) = (
    outcome = Ψ.outcome, 
    covariates = Ψ.outcome_extra_covariates, 
    confounders = Ψ.treatment_confounders,
    treatments = keys(Ψ.treatment_values)
    )

load_tmle_spec(file::Nothing) = (
    TMLE = TMLEE(
        models = TMLE.default_models(
            Q_binary = LogisticClassifier(lambda=0.),
            Q_continuous = LinearRegressor(),
            G = LogisticClassifier(lambda=0.)
        ),
        weighted = true, 
        ),
    )

function load_tmle_spec(file)
    include(abspath(file))
    return ESTIMATORS
end