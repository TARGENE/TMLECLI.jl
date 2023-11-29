
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
    make_summary(prefix; out="summary.json")

# Args

- `prefix`: Prefix to .hdf5 files to be used to create the summary file

# Options

- `-o, --out`: Ouptut JSON file
"""
@task function make_summary(prefix; output=JSONOutput(filename="summary.json"))
    dirname_, prefix_ = splitdir(prefix)
    dirname__ = dirname_ == "" ? "." : dirname_
    files = filter(
            x -> startswith(x, prefix_), 
            readdir(dirname__)
    )
    # Initialize JSON output
    initialize(output)
    # Write all but last batch
    for filename in files[1:end-1]        
        filepath = joinpath(dirname_, filename)
        jldopen(filepath) do io
            for batch_key in keys(io)
                update_file(output, io[batch_key])
            end
        end
    end
    # Write last batch
    filepath = joinpath(dirname_, files[end])
    jldopen(filepath) do io
        nkeys = length(keys(io))
        for (batch_index, batch_key) in enumerate(keys(io))
            finalize = batch_index == nkeys ? true : false
            update_file(output, io[batch_key], finalize=finalize)
        end
    end
    return 0
end