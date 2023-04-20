#####################################################################
#####                 CV ADAPTIVE FOLDS                          ####
#####################################################################

countuniques(v::AbstractVector) = [count(==(u), v) for u in unique(v)]
countuniques(table) = 
    countuniques([values(x) for x in Tables.namedtupleiterator(table)])

"""
    AdaptiveCV(cv::Union{CV, StratifiedCV})

Implements the rule of thum given here: https://www.youtube.com/watch?v=WYnjja8DKPg&t=4s
"""
mutable struct AdaptiveCV <: MLJ.ResamplingStrategy
    cv::Union{CV, StratifiedCV}
end


function MLJBase.train_test_pairs(cv::AdaptiveCV, rows, y)
    # Compute n-eff
    n = nrows(y)
    neff = 
        if scitype(first(y)) == MLJ.Continuous
            n
        else
            counts = countuniques(y)
            nrare = minimum(counts)
            min(n, 5*nrare)
        end

    # Compute number of folds
    nfolds = 
        if neff < 30
            neff
        elseif neff < 500
            20
        elseif neff < 5000
            10
        elseif neff < 10_000
            5
        else
            3
        end
    
    # Update the underlying n_folds
    adapted_cv = typeof(cv.cv)(nfolds=nfolds, shuffle=cv.cv.shuffle, rng=cv.cv.rng)
    
    return MLJBase.train_test_pairs(adapted_cv, rows, y)
end

#####################################################################
#####                       CSV OUTPUT                           ####
#####################################################################


csv_headers(;size=0) = DataFrame(
    PARAMETER_TYPE=Vector{String}(undef, size), 
    TREATMENTS=Vector{String}(undef, size), 
    CASE=Vector{String}(undef, size), 
    CONTROL=Vector{Union{Missing, String}}(undef, size), 
    TARGET=Vector{String}(undef, size), 
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
    length(Ψ.covariates) != 0 ? join(Ψ.covariates, join_string) : missing

function param_string(param::T) where T <: TMLE.Parameter
    str = string(T)
    return startswith(str, "TMLE.") ? str[6:end] : str
end

case(nt::NamedTuple) = nt.case
case(x) = x
case_string(Ψ; join_string="_&_") = join((case(x) for x in values(Ψ.treatment)), join_string)

control_string(t::Tuple{Vararg{<:NamedTuple}}; join_string="_&_") = 
    join((val.control for val in t), join_string)

control_string(t; join_string="_&_") = missing

control_string(Ψ::TMLE.Parameter; join_string="_&_") = 
    control_string(values(Ψ.treatment); join_string=join_string)

treatment_string(Ψ; join_string="_&_") = join(keys(Ψ.treatment), join_string)
confounders_string(Ψ; join_string="_&_") = join(Ψ.confounders, join_string)

function statistics_from_estimator(estimator)
    Ψ̂ = estimate(estimator)
    std = √(var(estimator))
    testresult = OneSampleTTest(estimator)
    pval = pvalue(testresult)
    l, u = confint(testresult)
    return (Ψ̂, std, pval, l, u)
end

function statistics_from_result(result::TMLE.TMLEResult)
    Ψ̂₀ = result.initial
    # TMLE stats
    tmle_stats = statistics_from_estimator(result.tmle) 
    # OneStep stats
    onestep_stats = statistics_from_estimator(result.onestep)
    return Ψ̂₀, tmle_stats, onestep_stats
end

statistics_from_result(result::Missing) = 
    missing, 
    (missing, missing, missing, missing, missing), 
    (missing, missing, missing, missing, missing)

function append_csv(filename, parameters, tmle_results, logs)
    data = csv_headers(size=size(tmle_results, 1))
    for (i, (Ψ, result, log)) in enumerate(zip(parameters, tmle_results, logs))
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
    CSV.write(filename, data, append=true, writeheader=!isfile(filename))
end


#####################################################################
#####                       JLD2 OUTPUT                          ####
#####################################################################

update_jld2_output(jld2_file::Nothing, parameters, partition, tmle_results, logs, dataset; pval_threshold=0.05) = nothing

function update_jld2_output(jld2_file::String, parameters, partition, tmle_results, logs, dataset; pval_threshold=0.05)
    if jld2_file !== nothing
        jldopen(jld2_file, "a+", compress=true) do io
        # Append only with results passing the threshold
            previous_variables = nothing
            sample_ids_idx = nothing
            partition_variables = [variables(parameters[index]) for index ∈ partition]

            for (partition_index, param_index) in enumerate(partition)
                Ψ = parameters[param_index]
                r = tmle_results[partition_index]
                log = logs[partition_index]
                if (r !== missing) && (pvalue(OneSampleZTest(r.tmle)) <= pval_threshold)
                    current_variables = partition_variables[partition_index]
                    if previous_variables != current_variables
                        sample_ids = TargetedEstimation.get_sample_ids(dataset, current_variables)
                        io["$param_index/sample_ids"] = sample_ids
                        sample_ids_idx = param_index
                    end
                    io["$param_index/result"] = r
                    io["$param_index/log"] = log
                    io["$param_index/sample_ids_idx"] = sample_ids_idx
                    io["$param_index/parameter"] = Ψ

                    previous_variables = current_variables
                end
            end
        end
    end
end

#####################################################################
#####                    Read Parameters                         ####
#####################################################################


function treatment_values(Ψ::Union{IATE, ATE}, treatment_names, treatment_types)
    return [(
        case = convert(treatment_types[tn], Ψ.treatment[tn].case), 
        control = convert(treatment_types[tn], Ψ.treatment[tn].control)
    ) 
        for tn in treatment_names]
end

treatment_values(Ψ::CM, treatment_names, treatment_types) = 
    [convert(treatment_types[tn], Ψ.treatment[tn]) for tn in treatment_names]

"""
    parameters_from_yaml(param_file, dataset)

Reads parameters from file and ensures that the parameters treatment in the config file
respect the treatment types in the dataset.
"""
function read_parameters(param_file, dataset)
    parameters = if any(endswith(param_file, ext) for ext in ("yaml", "yml"))
        parameters_from_yaml(param_file)
    else
        deserialize(param_file)
    end

    treatment_types = Dict()
    for index in eachindex(parameters)
        Ψ = parameters[index]
        treatment_names = keys(Ψ.treatment)
        for tn in treatment_names
            haskey(treatment_types, tn) ? nothing : treatment_types[tn] = eltype(dataset[!, tn])
        end
        new_treatment = NamedTuple{treatment_names}(
            treatment_values(Ψ, treatment_names, treatment_types)
        )
        parameters[index] = typeof(Ψ)(
            target = Ψ.target,
            treatment = new_treatment,
            confounders = Ψ.confounders,
            covariates = Ψ.covariates
        )
    end
    return parameters
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
    CSV.read(path, DataFrame, ntasks=1)

isbinary(col, dataset) = Set(unique(skipmissing(dataset[!, col]))) == Set([0, 1])


function make_categorical!(dataset, colname::Union{String, Symbol}; infer_ordered=false)
    ordered = false
    if infer_ordered
        ordered = eltype(dataset[!, colname]) <: Real
    end
    dataset[!, colname] = categorical(dataset[!, colname], ordered=ordered)
end

function make_categorical!(dataset, colnames; infer_ordered=false)
    for colname in colnames
        make_categorical!(dataset, colname;infer_ordered=infer_ordered)
    end
end

make_float!(dataset, colname::Union{String, Symbol}) = 
    dataset[!, colname] = float(dataset[!, colname])

function make_float!(dataset, colnames)
    for colname in colnames
        make_float!(dataset, colname)
    end
end

function coerce_types!(dataset, variables)
    # Treatment columns are converted to categorical
    make_categorical!(dataset, variables.treatments, infer_ordered=true)
    # Confounders and Covariates are converted to Float64
    make_float!(dataset, vcat(variables.confounders, variables.covariates))
    # Binary targets are converted to categorical
    make_categorical!(dataset, variables.binarytargets, infer_ordered=false)
end

variables(Ψ::TMLE.Parameter) = (
    target = Ψ.target, 
    covariates = Ψ.covariates, 
    confounders = Ψ.confounders,
    treatments = keys(Ψ.treatment)
    )

function variables(parameters::Vector{<:TMLE.Parameter}, dataset)
    treatments = Set{Symbol}()
    confounders = Set{Symbol}()
    covariates = Set{Symbol}()
    binarytargets = Set{Symbol}()
    continuoustargets = Set{Symbol}()
    for Ψ in parameters
        push!(treatments, keys(Ψ.treatment)...)
        push!(confounders, Ψ.confounders...)
        length(Ψ.covariates) > 0 && push!(covariates, Ψ.covariates...)
        isbinary(Ψ.target, dataset) ? push!(binarytargets, Ψ.target) : push!(continuoustargets, Ψ.target)
    end
    return (
        treatments=treatments, 
        confounders=confounders, 
        covariates=covariates, 
        binarytargets=binarytargets,
        continuoustargets
    )
end

