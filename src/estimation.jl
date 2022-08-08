
###############################################################################
####################         JLD2Saver Callback            ####################
###############################################################################

"""
A callback to save the TMLE estimation results to disk instead of keeping them in memory
"""
mutable struct JLD2Saver <: TMLE.Callback
    file::String
    save_machines::Bool
end

function TMLE.after_tmle(callback::JLD2Saver, report::TMLEReport, target_id::Int, query_id::Int)
    jldopen(callback.file, "a"; compress=true) do io
        group = haskey(io, "TMLEREPORTS") ? io["TMLEREPORTS"] : JLD2.Group(io, "TMLEREPORTS")
        group[string(target_id, "_", query_id)] = report
    end
end

function TMLE.after_fit(callback::JLD2Saver, mach::Machine, id::Symbol)
    if callback.save_machines
        jldopen(callback.file, "a"; compress=true) do io
            group = haskey(io, "MACHINES") ? io["MACHINES"] : JLD2.Group(io, "MACHINES")
            group[string(id)] = mach
        end
    end
end

function TMLE.finalize(callback::JLD2Saver, estimation_report::NamedTuple)
    jldopen(callback.file, "a"; compress=true) do io
        io["low_propensity_scores"] = estimation_report.low_propensity_scores
    end
    return estimation_report
end

###############################################################################
####################             Utilities                 ####################
###############################################################################

function sample_ids_per_phenotype(data, targets_columns)
    sample_ids = Dict()
    for colname in targets_columns
        sample_ids[colname] =
            dropmissing(data[!, ["SAMPLE_ID", colname]]).SAMPLE_ID
        
    end
    return sample_ids
end

convert_target(x, ::Type{Bool}) = categorical(x)
convert_target(x, ::Type{Real}) = x

function treatments_combinations(query)
    snps = keys(query.case)
    snp_levels = NamedTuple{snps}(
        [[getfield(query.control, snp), getfield(query.case, snp)] for snp in snps]
    )
    combinations = DataFrame(NamedTuple{snps}([[] for _ in snps]))
    for comb in reduce(vcat, Iterators.product(snp_levels...))
        push!(combinations, comb)
    end
    return combinations
end

function actualqueries(treatments, queries)
    unique_treatments = unique(dropmissing(treatments))
    actual_queries = []
    for query in queries
        required_treatments = treatments_combinations(query)
        joined = innerjoin(unique_treatments, required_treatments, on=names(treatments))
        if size(joined, 1) == size(required_treatments, 1)
            push!(actual_queries, query)
        else
            @warn string("Query: ", query.name, " will not be processed due to missing genotypic data.")
        end
    end
    return actual_queries
end

function preprocess(treatments, confounders, targets, target_type, queries)
    # Make sure data SAMPLE_ID types coincide
    treatments.SAMPLE_ID = string.(treatments.SAMPLE_ID)
    confounders.SAMPLE_ID = string.(confounders.SAMPLE_ID)
    targets.SAMPLE_ID = string.(targets.SAMPLE_ID)
    
    # columns names 
    treatments_columns = filter(!=("SAMPLE_ID"), names(treatments))
    confounders_columns = filter(!=("SAMPLE_ID"), names(confounders))
    targets_columns = filter(!=("SAMPLE_ID"), names(targets))

    # Join all elements together by SAMPLE_ID
    data = innerjoin(
            innerjoin(treatments, confounders, on=:SAMPLE_ID),
            targets,
            on=:SAMPLE_ID
            )

    # Drop missing values based on treatments and covariates 
    dropmissing!(data, vcat(treatments_columns, confounders_columns))

    # Retrieve T and convert to categorical data
    # The use of the query he is so that the order of the columns in both
    # T and queries are matching which is currently a technical requirement of the TMLE package
    T = DataFrame()
    for name in keys(first(queries).control)
        T[:, name] = categorical(data[!, name])
    end

    # Retrive sample_ids per phenotype
    sample_ids = sample_ids_per_phenotype(data, targets_columns)

    # Retrieve W
    W = DataFrame()
    for col in confounders_columns
        W[:, col] = convert(Vector{Float64}, data[!, col])
    end

    # Retrieve Y and convert to categorical if needed
    Y = DataFrame()
    for name in targets_columns
        Y[:, name] = convert_target(data[!, name], target_type)
    end

    return T, W, Y, sample_ids
end

function parse_queries(parameters_dicts)
    treatments = Tuple(parameters_dicts["Treatments"])
    queries = Query[]
    for param in parameters_dicts["Parameters"]
        case = NamedTuple{Symbol.(treatments)}([param[t]["case"] for t in treatments])
        control = NamedTuple{Symbol.(treatments)}([param[t]["control"] for t in treatments])
        push!(queries, TMLE.Query(case=case, control=control, name=param["name"]))
    end
    return queries
end

###############################################################################
####################           Main Function               ####################
###############################################################################

function tmle_run(parsed_args)
    v = parsed_args["verbosity"]
    target_type = parsed_args["target-type"] == "Real" ? Real : Bool
    parameters_dicts = YAML.load_file(parsed_args["parameters-file"])
    queries = parse_queries(parameters_dicts)
    treatment_cols = vcat("SAMPLE_ID", parameters_dicts["Treatments"][:])
    confounders_cols = haskey(parameters_dicts, "Confounders") ? vcat("SAMPLE_ID", parameters_dicts["Confounders"][:]) : nothing
    target_cols = haskey(parameters_dicts, "Targets") ? vcat("SAMPLE_ID", parameters_dicts["Targets"][:]) : nothing

    v >= 1 && @info "Loading data."
    # Load Treatment variables
    treatments = CSV.read(
        parsed_args["treatments"], 
        DataFrame; 
        select=treatment_cols
    )
    # Read Confounding variables
    confounders = CSV.read(
        parsed_args["confounders"], 
        DataFrame; 
        select=confounders_cols
    )
    # Load Target variables
    targets = CSV.read(
        parsed_args["targets"],
        DataFrame;
        select=target_cols
    )
    # Preprocessing
    v >= 1 && @info "Preprocessing data."
    treatments, confounders, targets, sample_ids = 
        preprocess(treatments, confounders, targets, target_type, queries)
    
    # Each genotype should have probability > 0
    queries = actualqueries(treatments, queries)
    if size(queries, 1) == 0
        jldopen(parsed_args["out"], "a") do io
            JLD2.Group(io, "TMLEREPORTS")
        end
        return 0
    end

    # Build estimator
    tmle = tmle_from_yaml(parsed_args["estimator-file"], queries, target_type)

    # tmle_run TMLE 
    v >= 1 && @info "TMLE Estimation."
    TMLE.fit(tmle, treatments, confounders, targets;
             verbosity=v-1, 
             cache=false,
             callbacks=[JLD2Saver(parsed_args["out"], parsed_args["save-full"])]
    )

    # Save sample_ids
    jldopen(parsed_args["out"], "a") do io
        io["SAMPLE_IDS"] = sample_ids
    end

    v >= 1 && @info "Done."
    return 0
end
