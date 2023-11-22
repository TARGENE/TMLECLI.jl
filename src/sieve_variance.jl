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
    OUTCOME=String[], 
    CONFOUNDERS=String[], 
    COVARIATES=Union{String, Missing}[], 
    TMLE_ESTIMATE=Float64[],
)

empty_sieve_output() = DataFrame(
    PARAMETER_TYPE=String[], 
    TREATMENTS=String[], 
    CASE=String[], 
    CONTROL=Union{String, Missing}[], 
    OUTCOME=String[], 
    CONFOUNDERS=String[], 
    COVARIATES=Union{String, Missing}[], 
    SIEVE_STD = Float64[],
    SIEVE_PVALUE = Float64[],
    SIEVE_LWB = Float64[],
    SIEVE_UPB = Float64[],
)

function push_sieveless!(output, Ψ, Ψ̂)
    target = string(Ψ.target)
    param_type = param_string(Ψ)
    treatments = treatment_string(Ψ)
    case = case_string(Ψ)
    control = control_string(Ψ)
    confounders = confounders_string(Ψ)
    covariates = covariates_string(Ψ)
    push!(output, (
        param_type, treatments, case, control, target, confounders, covariates, Ψ̂
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
    hdf5files = sort([joinpath(dirname_, x) for x in hdf5files])

    influence_curves = Vector{Float32}[]
    n_obs = Int[]
    tmle_results = []
    for hdf5file in hdf5files
        jldopen(hdf5file) do io
            # templateΨs = io["parameters"]
            # results = io["results"]
            for key in keys(io)
                result_group = io[key]
                tmleresult = first(io[key]["result"])
                if size(tmleresult.IC, 1) > 0
                    sample_ids = haskey(result_group, "sample_ids") ? result_group["sample_ids"] :
                        io[string(result_group["sample_ids_idx"])]["sample_ids"]
                    sample_ids = string.(sample_ids)

                    push!(influence_curves, align_ic(tmleresult.IC, sample_ids, grm_ids))
                    push!(n_obs, size(sample_ids, 1))
                    push!(tmle_results, tmleresult)
                end
            end
        end
    end
    influence_curves = length(influence_curves) > 0 ? reduce(vcat, transpose(influence_curves)) : Matrix{Float32}(undef, 0, 0)
    return tmle_results, influence_curves, n_obs
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

function save_results(outprefix, results, τs, variances)
    TMLE.write_json(string(outprefix, ".json"), results)
    jldopen(string(outprefix, ".hdf5"), "w") do io
        io["taus"] = τs
        io["variances"] = variances
    end
end

corrected_stderrors(variances) =
    sqrt.(view(maximum(variances, dims=1), 1, :))

function update_with_sieve_estimate!(results, stds)
    for index in eachindex(results)
        old = results[index]
        results[index] = typeof(old)(
            old.estimand,
            old.estimate,
            convert(Float64, stds[index]),
            old.n,
            Float64[]
        )
    end
end

"""
    sieve_variance_plateau(input_prefix;
        output_prefix="svp",
        grm_prefix="GRM",
        verbosity=0, 
        n_estimators=10, 
        max_tau=0.8
    )

Sieve Variance Plateau CLI.

# Args

- `input-prefix`: Input prefix to HDF5 files generated by the tmle CLI.

# Options

- `-o, --output-prefix`: Output prefix.
- `-g, --grm-prefix`: Prefix to the aggregated GRM.
- `-v, --verbosity`: Verbosity level.
- `-n, --n_estimators`: Number of variance estimators to build for each estimate. 
- `-m, --max_tau`: Maximum distance between any two individuals.
"""
@cast function sieve_variance_plateau(input_prefix;
    output_prefix="svp",
    grm_prefix="GRM",
    verbosity=0, 
    n_estimators=10, 
    max_tau=0.8
    )
    τs = default_τs(n_estimators;max_τ=max_tau)
    grm, grm_ids = readGRM(grm_prefix)
    verbosity > 0 && @info "Preparing work list."
    results, influence_curves, n_obs = build_work_list(input_prefix, grm_ids)

    if length(influence_curves) > 0
        verbosity > 0 && @info "Computing variance estimates."
        variances = compute_variances(influence_curves, grm, τs, n_obs)
        std_errors = corrected_stderrors(variances)
        update_with_sieve_estimate!(results, std_errors)
    else
        variances = Float32[]
    end
    save_results(output_prefix, results, τs, variances)

    verbosity > 0 && @info "Done."
    return 0
end
