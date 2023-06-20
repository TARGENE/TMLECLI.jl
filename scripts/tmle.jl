using ArgParse
using TargetedEstimation

function parse_commandline()
    s = ArgParseSettings(
        description = "Targeted Learning estimation",
        commands_are_required = false,
        version = "0.2",
        add_version = true)

    @add_arg_table s begin
        "data"
            help = "Path to dataset file (.csv|.arrow)"
            required = true
        "param-file"
            help = "A file (.yaml|.bin) listing all parameters to estimate."
            required = true
        "csv-out"
            help = "Path to output `.csv` file"
            required = true
        "--estimator-file"
            help = "A file (.jl) describing the tmle estimator to use, README.md"
            arg_type= String
            required = false
        "--hdf5-out"
            help = "If the influence curves also need to be stored (see also: --pval-threshold)"
            arg_type = String
            default = nothing
        "--pval-threshold"
            help = "Only those parameters passing the threshold will have their influence curve saved."
            default = 1.
            arg_type = Float64
        "--chunksize"
            help = "Results will be appended to outfiles every chunk"
            default = 100
            arg_type = Int
        "--verbosity", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

tmle_estimation(parsed_args)


