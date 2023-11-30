
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

"""
    make_summary(
        prefix; 
        outputs=Outputs(json=JSONOutput(filename="summary.json"))
    )

Combines multiple TMLE .hdf5 output files in a single file. Multiple formats can be output at once.

# Args

- `prefix`: Prefix to .hdf5 files to be used to create the summary file

# Options

- `-o, --outputs`: Ouptuts configuration.
"""
@cast function make_summary(
    prefix; 
    outputs=Outputs(json=JSONOutput(filename="summary.json"))
    )
    
    # Initialize output files
    initialize(outputs)
    actual_outputs = [getfield(outputs, field) for field ∈ fieldnames(Outputs) 
        if getfield(outputs, field).filename !== nothing]

    # Get all input .hdf5 files
    dirname_, prefix_ = splitdir(prefix)
    dirname__ = dirname_ == "" ? "." : dirname_
    files = sort(filter(
            x -> startswith(x, prefix_), 
            readdir(dirname__)
    ))
    nfiles = length(files)

    # Write to files
    for (file_index, filename) in enumerate(files)
        filepath = joinpath(dirname_, filename)
        jldopen(filepath) do io
            batch_keys = collect(keys(io))
            nbatches = length(batch_keys)
            for (batch_index, batch_key) in enumerate(batch_keys)
                results = io[batch_key]
                finalize = file_index == nfiles && batch_index == nbatches
                for output in actual_outputs
                    update_file(output, results; finalize=finalize)
                end
            end
        end
    end

    return 0
end