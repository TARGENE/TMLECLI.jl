GRMIDs(file) = CSV.File(file, 
                        header=["FAMILY_ID", "SAMPLE_ID"], 
                        select=["SAMPLE_ID"],
                        types=String) |> DataFrame

function readGRM(prefix)
    ids = GRMIDs(string(prefix, ".id"))
    n = size(ids, 1)
    grm_size = n*(n + 1) ÷ 2
    GRM = mmap(string(prefix, ".bin"), Vector{Float32}, grm_size)

    return GRM, ids
end

function align_ic(ic, sample_ids, grm_ids)
    leftjoin!(
        grm_ids, 
        DataFrame(IC=ic, SAMPLE_ID=sample_ids),
        on=:SAMPLE_ID
    )
    aligned_ic = grm_ids.IC
    select!(grm_ids, Not(:IC))
    return coalesce.(aligned_ic, 0)
end

sieve_dataframe() = DataFrame(
    PARAMETER_TYPE=String[], 
    TREATMENTS=String[], 
    CASE=String[], 
    CONTROL=Union{String, Missing}[], 
    TARGET=String[], 
    CONFOUNDERS=String[], 
    COVARIATES=Union{String, Missing}[], 
    ESTIMATE=Float64[],
)

function push_sieveless!(output, Ψ, Ψ̂, target)
    param_type = param_string(Ψ)
    treatments = treatment_string(Ψ)
    case = case_string(Ψ)
    control = control_string(Ψ)
    confounders = confounders_string(Ψ)
    covariates = covariates_string(Ψ)
    push!(output, (
        param_type, treatments, case, control, restore_slash(target), confounders, covariates, Ψ̂
    ))
end

"""
    bit_distances(sample_grm, nτs)

Returns a matrix of shape (n_samples,nτs) where n_samples is the 
size of sample_grm.
The sample_grm comes from the gcta software. 
The process is as follows:
- Round between -1 and 1 as some elements may be beyond that limit
- Take 1 - this value to convert this quantity to a distance
- For each τ return if the distance between individuals is less or equal than τ
"""
function bit_distances(sample_grm, τs)
    distances = 1 .-  max.(min.(sample_grm, 1), -1)
    return convert(Matrix{Float32}, permutedims(distances) .<= τs)
end


default_τs(nτs;max_τ=2) = Float32[max_τ*(i-1)/(nτs-1) for i in 1:nτs]


function build_work_list(prefix, grm_ids)
    dirname_, prefix_ = splitdir(prefix)
    dirname__ = dirname_ == "" ? "." : dirname_
    hdf5files = filter(
            x -> startswith(x, prefix_) && endswith(x, ".hdf5"), 
            readdir(dirname__)
    )
    hdf5files = [joinpath(dirname_, x) for x in hdf5files]

    influence_curves = Vector{Float32}[]
    n_obs = Int[]
    sieve_df = sieve_dataframe()
    for hdf5file in hdf5files
        jldopen(hdf5file) do io
            templateΨs = io["parameters"]
            results = io["results"]
            for target in keys(results)
                targetresults = results[target]
                sample_ids = string.(targetresults["sample_ids"])
                for index in eachindex(targetresults["tmle_results"])
                    templateΨ = templateΨs[index]
                    tmleresult = targetresults["tmle_results"][index]
                    Ψ̂ = TMLE.estimate(tmleresult)
                    push!(influence_curves, align_ic(tmleresult.IC, sample_ids, grm_ids))
                    push!(n_obs, size(sample_ids, 1))
                    push_sieveless!(sieve_df, templateΨ, Ψ̂, target)
                end
            end
        end
    end
    return sieve_df, reduce(vcat, transpose(influence_curves)), n_obs
end


"""
    normalize(variances, n_observations)

Divides the variance estimates by the effective number of observations 
used for each phenotype at estimation time.
"""
normalize!(variances, n_observations) = 
    variances ./= permutedims(n_observations)


"""
    aggregate_variances(influence_curves, indicator, sample)

This function computes the sum for a single index i, see also `compute_variances`.
As the GRM is symetric it is performed as : 
    2 times off-diagonal elements with j < i + diagonal term 
and this for all τs. Some diagonal elements are not equal to 1 while in theory they should be,
this is an artefact of the GRM. We thus always include diagonal elements in the aggregate 
whatever the value of the indicator.
"""
function aggregate_variances(influence_curves, indicator, sample)
    @views begin
        D_off_diag = transpose(influence_curves[:, 1:sample-1])
        D_diag = transpose(influence_curves[:, sample])
        return D_diag .* (2indicator[:, 1:sample-1] * D_off_diag .+ D_diag)
    end
end


"""
    compute_variances(influence_curves, nτs, grm_files)

An overall variance estimate for a distance function d, a threshold τ 
and influence curve D is given by:
            σ̂ = 1/n ∑ᵢ∑ⱼ1(dᵢⱼ ≤ τ)DᵢDⱼ
              = 1/n 2∑ᵢDᵢ∑ⱼ<ᵢ1(dᵢⱼ ≤ τ)DᵢDⱼ + 1(dᵢᵢ ≤ τ)DᵢDᵢ

This function computes those variance estimates at each τ for all phenotypes
and queries.

# Arguments:
- influence_curves: Array of size (n_samples, n_queries, n_phenotypes)
- grm: Vector containing the lower elements of the GRM ie: n(n+1)/2 elements
- τs: 1 row matrix containing the distance thresholds between individuals
- n_obs: Vector of size (n_phenotypes,), containing the number of effective 
observations used during estimation

# Returns:
- variances: An Array of size (nτs, n_curves) where n_curves is the number of influence curves.
"""
function compute_variances(influence_curves, grm, τs, n_obs)
    n_curves, n_samples = size(influence_curves)
    variances = zeros(Float32, length(τs), n_curves)
    start_idx = 1
    for sample in 1:n_samples
        # lower diagonal of the GRM are stored in a single vector 
        # that are accessed one row at a time
        end_idx = start_idx + sample - 1
        sample_grm = view(grm, start_idx:end_idx)
        indicator = bit_distances(sample_grm, τs)
        variances .+= aggregate_variances(influence_curves, indicator, sample)
        start_idx = end_idx + 1
    end
    normalize!(variances, n_obs)
    return variances
end


function grm_rows_bounds(n_samples)
    bounds = Pair{Int, Int}[]
    start_idx = 1
    for sample in 1:n_samples
        # lower diagonal of the GRM are stored in a single vector 
        # that are accessed one row at a time
        end_idx = start_idx + sample - 1
        push!(bounds, start_idx => end_idx)
        start_idx = end_idx + 1
    end
    return bounds
end


function save_results(outprefix, output, τs, variances)
    CSV.write(string(outprefix, ".csv"), output)
    jldopen(string(outprefix, ".hdf5"), "w") do io
        io["taus"] = τs
        io["variances"] = variances
    end
end


corrected_stderrors(variances, n_obs) =
    sqrt.(view(maximum(variances, dims=1), 1, :) ./ n_obs)

function update_sieve_df!(df, stds, n_obs)
    n = size(stds, 1)
    df.SIEVE_STD = Vector{Float64}(undef, n)
    df.SIEVE_PVALUE = Vector{Float64}(undef, n)
    df.SIEVE_LWB = Vector{Float64}(undef, n)
    df.SIEVE_UPB = Vector{Float64}(undef, n)

    for index in 1:n
        std = stds[index]
        estimate = df.ESTIMATE[index]
        testresult = OneSampleZTest(estimate, std, n_obs[index])
        lwb, upb = confint(testresult)
        df.SIEVE_STD[index] = std
        df.SIEVE_PVALUE[index] = pvalue(testresult)
        df.SIEVE_LWB[index] = lwb
        df.SIEVE_UPB[index] = upb
    end
end

function sieve_variance_plateau(parsed_args)
    prefix = parsed_args["prefix"]
    outprefix = parsed_args["out-prefix"]
    verbosity = parsed_args["verbosity"]

    τs = default_τs(parsed_args["nb-estimators"];max_τ=parsed_args["max-tau"])
    grm, grm_ids = readGRM(parsed_args["grm-prefix"])
    verbosity > 0 && @info "Preparing work list."
    sieve_df, influence_curves, n_obs = build_work_list(prefix, grm_ids)

    verbosity > 0 && @info "Computing variance estimates."

    variances = compute_variances(influence_curves, grm, τs, n_obs)
    std_errors = corrected_stderrors(variances, n_obs)
    update_sieve_df!(sieve_df, std_errors, n_obs)
    save_results(outprefix, sieve_df, τs, variances)

    verbosity > 0 && @info "Done."
    return 0
end
