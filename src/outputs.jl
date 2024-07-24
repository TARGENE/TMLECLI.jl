#####################################################################
#####                       JSON OUTPUT                          ####
#####################################################################

write_json(filename, results) = open(filename, "w") do io
    JSON.print(io, TMLE.to_dict(results))
end

initialize_json(filename::Nothing) = 0

initialize_json(filename::String) = open(filename, "w") do io
    print(io, '[')
end

finalize_json(filename::Nothing) = 0

function finalize_json(filename::String)
    open(filename, "a") do io
        skip(io, -1) # get rid of the last comma which JSON doesn't allow
        print(io, ']')
    end
end

update_json(filename::Nothing, result) = 0

function update_json(filename, results)
    open(filename, "a") do io
        for result in results
            JSON.print(io, TMLE.to_dict(result))
            print(io, ',')
        end
    end
end

#####################################################################
#####                       HDF5 OUTPUT                          ####
#####################################################################

write_hdf5(filename, results)= jldopen(io -> io["results"] = results, filename, "w")

update_hdf5(filename::Nothing, results) = 0

function update_hdf5(filename, results)
    jldopen(filename, "a+") do io
        batches_keys = keys(io)
        latest_index = isempty(batches_keys) ? 0 : maximum(parse(Int, split(key, "_")[2]) for key in batches_keys)
        io[string("Batch_", latest_index + 1)] = results
    end
end

#####################################################################
#####                        JLS OUTPUT                          ####
#####################################################################

write_jls(filename, results) = serialize(filename, results)

update_jls(filename::Nothing, results) = 0

function update_jls(filename, results)
    open(filename, "a") do io
        for result in results
            serialize(io, result)
        end
    end
end

#####################################################################
#####                         OUTPUTS                            ####
#####################################################################

struct Outputs
    json::Union{String, Nothing}
    hdf5::Union{String, Nothing}
    jls::Union{String, Nothing}
end

Outputs(;json=nothing, hdf5=nothing, jls=nothing) = Outputs(json, hdf5, jls)

function write(outputs::Outputs, results)
    # Append JSON Output
    write_json(outputs.json, results)
    # Append JLS Output
    write_jls(outputs.jls, results)
    # Append HDF5 Output
    write_hdf5(outputs.hdf5, results)
end

clean_output_file(::Nothing) = 0

function clean_output_file(file)
    if isfile(file)
        rm(file)
    end
end

"""
    initialize(output::Outputs)

Initializes all outputs in output.
"""
function initialize(outputs::Outputs)
    for file in (outputs.json, outputs.hdf5, outputs.jls)
        clean_output_file(file)
    end
    # Initialize JSON file if specified
    initialize_json(outputs.json)
end

function update(outputs::Outputs, results)
    # Append JSON Output
    update_json(outputs.json, results)
    # Append JLS Output
    update_jls(outputs.jls, results)
    # Append HDF5 Output
    update_hdf5(outputs.hdf5, results)
end

finalize(outputs::Outputs) = finalize_json(outputs.json)

function add_sample_ids_to_results(results, dataset)
    sample_ids = get_sample_ids(dataset, results)
    return [(result..., SAMPLE_IDS=s_ids) for (result, s_ids) in zip(results, sample_ids)]
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