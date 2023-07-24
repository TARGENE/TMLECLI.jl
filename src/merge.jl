
function files_matching_prefix_and_suffix(prefix, suffix)
    dirname_, prefix_ = splitdir(prefix)
    dirname__ = dirname_ == "" ? "." : dirname_
    files = filter(
            x -> startswith(x, prefix_) && endswith(x, suffix), 
            readdir(dirname__)
    )
    return [joinpath(dirname_, x) for x in files]
end

read_output_with_types(file) = 
    CSV.read(file, DataFrame, types=Dict(key => String for key in joining_keys()))

function load_csv_files(data, files)
    for file in files
        new_data = read_output_with_types(file)
        if size(new_data, 1) > 0 
            data = vcat(data, new_data)
        end
    end
    return data
end

joining_keys() = ["PARAMETER_TYPE", "TREATMENTS", "CASE", "CONTROL", "TARGET", "CONFOUNDERS", "COVARIATES"]

function merge_csv_files(parsed_args)
    tmle_files = files_matching_prefix_and_suffix(
        parsed_args["tmle-prefix"],
        ".csv"
    )
    # Load tmle data
    data = load_csv_files(empty_tmle_output(), tmle_files)
    # Load sieve data
    sieveprefix = parsed_args["sieve-prefix"]
    if sieveprefix !== nothing
        sieve_files = files_matching_prefix_and_suffix(
            parsed_args["sieve-prefix"],
            ".csv"
        )
        sieve_data = load_csv_files(empty_sieve_output(), sieve_files)
        if size(sieve_data, 1) > 0
            data = leftjoin(data, sieve_data, on=joining_keys(), matchmissing=:equal)
        end
    end

    # Pvalue Adjustment by Target
    for gp in groupby(data, :TARGET)
        gp.TRAIT_ADJUSTED_TMLE_PVALUE = gp[:, :TMLE_PVALUE]
        pvalues = collect(skipmissing(gp.TMLE_PVALUE))
        if length(pvalues) > 0
            adjusted_pvalues = adjust(pvalues, BenjaminiHochberg())
            adjusted_pval_index = 1
            for index in eachindex(gp.TRAIT_ADJUSTED_TMLE_PVALUE)
                gp.TRAIT_ADJUSTED_TMLE_PVALUE[index] === missing && continue
                gp.TRAIT_ADJUSTED_TMLE_PVALUE[index] = adjusted_pvalues[adjusted_pval_index]
                adjusted_pval_index += 1
            end
        end
    end

    # Write to output file
    CSV.write(parsed_args["out"], data)

    return 0
end