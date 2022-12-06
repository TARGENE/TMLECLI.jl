struct SaveFull{x} end

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


csv_headers() = DataFrame(
    PARAMETER_TYPE=[], 
    TREATMENTS=[], 
    CASE=[], 
    CONTROL=[], 
    TARGET=[], 
    CONFOUNDERS=[], 
    COVARIATES=[], 
    INITIAL_ESTIMATE=[], 
    ESTIMATE=[],
    STD=[],
    PVALUE=[],
    LWB=[],
    UPB=[],
    LOG=[]
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

initialize_outfile(io, gpd_parameters) =
    CSV.write(io, csv_headers())


function statistics_from_result(result)
    Ψ̂ = estimate(result)
    std = √(var(result))
    testresult = OneSampleTTest(result)
    pval = pvalue(testresult)
    lw, up = confint(testresult)
    return Ψ̂, std, pval, lw, up
end

statistics_from_result(result::Missing) = 
    missing, missing, missing, missing, missing

function write_target_results(io, target_parameters, tmle_results, initial_estimates, sample_ids, logs)
    data = csv_headers()
    for (Ψ, result, Ψ̂₀, log) in zip(target_parameters.PARAMETER, tmle_results, initial_estimates, logs)
        param_type = param_string(Ψ)
        treatments = treatment_string(Ψ)
        case = case_string(Ψ)
        control = control_string(Ψ)
        confounders = confounders_string(Ψ)
        covariates = covariates_string(Ψ)
        Ψ̂, std, pval, lw, up = statistics_from_result(result)
        row = (param_type, treatments, case, control, string(Ψ.target), confounders, covariates, Ψ̂₀, Ψ̂, std, pval, lw, up, log)
        push!(data, row)
    end
    CSV.write(io, data, append=true)
end

open_io(f, file, save_full::SaveFull{false}, mode="w", args...; kwargs...) =
    open(f, file, mode, args...; kwargs...)

#####################################################################
#####                       JLD2 OUTPUT                          ####
#####################################################################
no_slash(x) = replace(string(x), "/" => "_OR_")

initialize_outfile(io::JLD2.JLDFile, gpd_parameters) =
    io["parameters"] = first(gpd_parameters)[!, "PARAMETER"]

function write_target_results(io::JLD2.JLDFile, target_parameters, tmle_results, initial_estimates, sample_ids, logs)
    target = no_slash(target_parameters[1, :TARGET])
    io["results/$target/initial_estimates"] = initial_estimates
    io["results/$target/tmle_results"] = tmle_results
    io["results/$target/sample_ids"] = sample_ids
    io["results/$target/logs"] = logs
end


open_io(f, file, save_full::SaveFull{true}, mode="w", args...; compress=true, kwargs...) =
    jldopen(f, file, mode, args...; compress=compress, kwargs...)


#####################################################################
#####                 ADDITIONAL METHODS                         ####
#####################################################################

get_non_target_columns(parameter) =
    vcat(keys(parameter.treatment)..., parameter.confounders, parameter.covariates)


get_sample_ids(data, targets_columns, save_full::SaveFull{true}) = dropmissing(data[!, [:SAMPLE_ID, targets_columns...]]).SAMPLE_ID

get_sample_ids(data, targets_columns, save_full::SaveFull{false}) = nothing

"""
    instantiate_dataset(path::String)

Returns a DataFrame wrapper around a dataset, either in CSV format.
"""
function instantiate_dataset(path::String)
        return CSV.read(path, DataFrame)
end

isbinarytarget(y::AbstractVector) = Set(unique(skipmissing(y))) == Set([0, 1])

function nuisance_spec_from_target(tmle_spec, isbinary)
    Q_spec = isbinary ? tmle_spec.Q_binary : tmle_spec.Q_continuous
    return NuisanceSpec(Q_spec, tmle_spec.G)
end

maybe_categorical(v) = categorical(v)
maybe_categorical(v::CategoricalArray) = v

make_categorical!(dataset, colname::Union{String, Symbol}, isbinary::Bool) =
    isbinary ? dataset[!, colname] = maybe_categorical(dataset[!, colname]) : nothing

function make_categorical!(dataset, colnames::Tuple, isbinary::Bool)
    for colname in colnames
        make_categorical!(dataset, colname, isbinary)
    end
end

tmle_run!(cache::Nothing, Ψ, η_spec, dataset; verbosity=1, threshold=1e-8) = 
    tmle(Ψ, η_spec, dataset; verbosity=verbosity, threshold=threshold)

tmle_run!(cache, Ψ, η_spec, dataset; verbosity=1, threshold=1e-8) = 
    tmle!(cache, Ψ, η_spec; verbosity=verbosity, threshold=threshold)


TMLE.estimate(e::Missing) = missing