#####################################################################
#####                       JSON OUTPUT                          ####
#####################################################################

@option struct JSONOutput
    filename::Union{Nothing, String} = nothing
    pval_threshold::Union{Nothing, Float64} = nothing
end

initialize(output::JSONOutput) = initialize_json(output.filename)

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
#####                       HDF5 OUTPUT                          ####
#####################################################################

@option struct HDF5Output
    filename::Union{Nothing, String} = nothing
    pval_threshold::Union{Nothing, Float64} = nothing
    sample_ids::Bool = false
    compress::Bool = false
end

function update_file(output::HDF5Output, results, dataset)
    output.filename === nothing && return
    results = post_process(results, dataset, output.pval_threshold, output.sample_ids)
    jldopen(output.filename, "a+", compress=output.compress) do io
        batches_keys = keys(io)
        latest_index = isempty(batches_keys) ? 0 : maximum(parse(Int, split(key, "_")[2]) for key in batches_keys)
        io[string("Batch_", latest_index + 1)] = results
    end
end

#####################################################################
#####                        JLS OUTPUT                          ####
#####################################################################

@option struct JLSOutput
    filename::Union{Nothing, String} = nothing
    pval_threshold::Union{Nothing, Float64} = nothing
    sample_ids::Bool = false
end

function update_file(output::JLSOutput, results, dataset)
    output.filename === nothing && return
    results = post_process(results, dataset, output.pval_threshold, output.sample_ids)

    open(output.filename, "a") do io
        for result in results
            serialize(io, result)
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
#####                         OUTPUTS                            ####
#####################################################################

@option struct Outputs
    json::JSONOutput = JSONOutput()
    hdf5::HDF5Output = HDF5Output()
    jls::JLSOutput   = JLSOutput()
    std::Bool        = false
end

function initialize(outputs::Outputs)
    initialize(outputs.json)
end

function post_process(results, dataset, pval_threshold, save_sample_ids)
    results = [TMLE.emptyIC(result, pval_threshold) for result ∈ results]
    if save_sample_ids
        sample_ids = get_sample_ids(dataset, results)
        results = [(result..., SAMPLE_IDS=s_ids) for (result, s_ids) in zip(results, sample_ids)]
    end
    return results
end

sample_ids_from_variables(dataset, variables) = dropmissing(dataset[!, [:SAMPLE_ID, variables...]]).SAMPLE_ID

function get_sample_ids(dataset, results)
    previous_variables = nothing
    sample_ids = []
    current_ref_id = 0
    for (index, result) in enumerate(results)
        current_variables = variables(first(result).estimand)
        if previous_variables != current_variables
            push!(sample_ids, sample_ids_from_variables(dataset, current_variables))
            current_ref_id = index
            previous_variables = current_variables
        else
            push!(sample_ids, current_ref_id)
        end
    end
    return sample_ids
end