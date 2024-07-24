append_results_from_key!(results, io, key) = append!(results, io[key])

function append_results_from_file!(results, file)
    jldopen(file) do io
        for key in keys(io)
            append_results_from_key!(results, io, key)
        end
    end
end

function read_results_from_files(files)
    results = []
    for file in files
        append_results_from_file!(results, file)
    end
    return results
end

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
function make_summary(
    prefix::String;
    outputs::Outputs=Outputs()
    )
    # Get all input .hdf5 files
    dirname_, prefix_ = splitdir(prefix)
    dirname__ = dirname_ == "" ? "." : dirname_
    files = sort(filter(
            x -> startswith(x, prefix_), 
            readdir(dirname__)
    ))
    # Combine all results in a single vector
    results = read_results_from_files(joinpath.(dirname_, files))
    # Write to outputs
    write(outputs, results)

    return 0
end