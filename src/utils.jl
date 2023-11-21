#####################################################################
#####                       JSON OUTPUT                          ####
#####################################################################

initialize_json(filename::Nothing) = nothing

initialize_json(filename::String) = open(filename, "w") do io
    print(io, '[')
end

function update_file(output::JSONOutput, results; finalize=false)
    output.filename === nothing && return
    open(output.filename, "a") do io
        for result in results
            result = TMLE.emptyIC(result, output.pval_threshold)
            JSON.print(io, TMLE.to_dict(result))
            print(io, ',')
        end
        if finalize
            skip(io, -1) # get rid of the last comma which JSON doesn't allow
            print(io, ']')
        end
    end
end

#####################################################################
#####                       STD OUTPUT                          ####
#####################################################################

function update_file(doprint, results, partition)
    if doprint
        mimetext = MIME"text/plain"()
        index = 1
        for (result, estimand_index) in zip(results, partition)
            show(stdout, mimetext, string("⋆⋆⋆ Estimand ", estimand_index, " ⋆⋆⋆"))
            println(stdout)
            show(stdout, mimetext, first(result).estimand)
            for (key, val) ∈ zip(keys(result), result)
                show(stdout, mimetext, string("→ Estimation Result From: ", key, ))
                println(stdout)
                show(stdout, mimetext, val)
                index += 1
            end
        end
    end
end

#####################################################################
#####                       JLD2 OUTPUT                          ####
#####################################################################


function update_file(output::HDF5Output, partition, results, dataset)
    output.filename === nothing && return

    jldopen(output.filename, "a+", compress=true) do io
        previous_variables = nothing
        sample_ids_idx = nothing
        for (partition_index, param_index) in enumerate(partition)
            estimator_results = TMLE.emptyIC(results[partition_index], output.pval_threshold)
            current_variables = variables(first(estimator_results).estimand)
            if previous_variables != current_variables
                sample_ids = TargetedEstimation.get_sample_ids(dataset, current_variables)
                io["$param_index/sample_ids"] = sample_ids
                sample_ids_idx = param_index
            end
            io["$param_index/result"] = estimator_results
            io["$param_index/sample_ids_idx"] = sample_ids_idx

            previous_variables = current_variables
        end
    end
end

#####################################################################
#####                    Read TMLE Estimands Configuration                         ####
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

read_method(extension) = extension == ".json" ? read_json : read_yaml

"""
    proofread_estimands(param_file, dataset)

Reads estimands from file and ensures that the treatment values in the config file
respects the treatment types in the dataset.
"""
function proofread_estimands(filename, dataset)
    extension = filename[findlast(isequal('.'), filename):end]
    config = read_method(extension)(filename)
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

TMLE.emptyIC(result::FailedEstimation) = result

TMLE.emptyIC(result::FailedEstimation, pval_threshold::Float64) = result

TMLE.emptyIC(result::NamedTuple{names}, pval_threshold::Nothing) where names = 
    NamedTuple{names}([TMLE.emptyIC(r) for r in result])

TMLE.emptyIC(result::NamedTuple{names}, pval_threshold::Float64) where names =
    NamedTuple{names}([TMLE.emptyIC(r, pval_threshold) for r in result])

function TMLE.emptyIC(result, pval_threshold::Float64)
    pval = pvalue(OneSampleZTest(result))
    return pval < pval_threshold ? result : TMLE.emptyIC(result)
end


get_sample_ids(data, variables) = dropmissing(data[!, [:SAMPLE_ID, variables...]]).SAMPLE_ID


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

variables(Ψ::TMLE.Estimand) = Set([
    Ψ.outcome,
    keys(Ψ.treatment_values)...,
    Ψ.outcome_extra_covariates..., 
    Iterators.flatten(values(Ψ.treatment_confounders))...
    ])

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

TMLE.to_dict(nt::NamedTuple{names, <:Tuple{Vararg{TMLE.EICEstimate}}}) where names = 
    Dict(key => TMLE.to_dict(val) for (key, val) ∈ zip(keys(nt), nt))

TMLE.to_dict(nt::NamedTuple{names, <:Tuple{Vararg{FailedEstimation}}}) where names = 
    Dict(key => TMLE.to_dict(val) for (key, val) ∈ zip(keys(nt), nt))