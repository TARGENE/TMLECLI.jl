FileExistsError(filename) = ArgumentError(string("File ", filename, " already exists."))

check_file_exists(filename::Nothing) = nothing
check_file_exists(filename) = !isfile(filename) || throw(FileExistsError(filename))

Base.tryparse(::Type{Union{String, Nothing}}, x::AbstractString) = x
Base.tryparse(::Type{Union{Float64, Nothing}}, x::AbstractString) = tryparse(Float64, x)
Base.tryparse(::Type{Union{T, Nothing}}, x::Nothing) where T = nothing

"""
    initialize(output)

Default intialization procedure only checks that file does not exist.
"""
initialize(output) = check_file_exists(output.filename)

#####################################################################
#####                       JSON OUTPUT                          ####
#####################################################################

@option struct JSONOutput
    filename::Union{Nothing, String} = nothing
    pval_threshold::Union{Nothing, Float64} = nothing
end

"""
    initialize(output::JSONOutput)

Checks that file does not exist and inialize the json file
"""
function initialize(output::JSONOutput)
    check_file_exists(output.filename)
    initialize_json(output.filename)
end

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

function update_file(output::HDF5Output, results; finalize=false)
    jldopen(output.filename, "a+", compress=output.compress) do io
        batches_keys = keys(io)
        latest_index = isempty(batches_keys) ? 0 : maximum(parse(Int, split(key, "_")[2]) for key in batches_keys)
        io[string("Batch_", latest_index + 1)] = results
    end
end

function update_file(output::HDF5Output, results, dataset)
    output.filename === nothing && return
    results = post_process(results, dataset, output.pval_threshold, output.sample_ids)
    update_file(output, results)
end

#####################################################################
#####                        JLS OUTPUT                          ####
#####################################################################

@option struct JLSOutput
    filename::Union{Nothing, String} = nothing
    pval_threshold::Union{Nothing, Float64} = nothing
    sample_ids::Bool = false
end

function update_file(output::JLSOutput, results; finalize=false)
    open(output.filename, "a") do io
        for result in results
            serialize(io, result)
        end
    end
end

function update_file(output::JLSOutput, results, dataset)
    output.filename === nothing && return
    results = post_process(results, dataset, output.pval_threshold, output.sample_ids)
    update_file(output, results)
end


#####################################################################
#####                         OUTPUTS                            ####
#####################################################################

@option struct Outputs
    json::JSONOutput = JSONOutput()
    hdf5::HDF5Output = HDF5Output()
    jls::JLSOutput   = JLSOutput()
end

"""
    initialize(output::Outputs)

Initializes all outputs in output.
"""
function initialize(outputs::Outputs)
    initialize(outputs.json)
    initialize(outputs.jls)
    initialize(outputs.hdf5)
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