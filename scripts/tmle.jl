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
            help = "Path to dataset file (.csv)"
            required = true
        "param-file"
            help = "A file (.yaml format) see README.md"
            required = true
        "estimator-file"
            help = "A file (.yaml format) describing the tmle estimator to use, README.md"
            required = true
        "outprefix"
            help = "Prefix to output files. A `.csv` file is always generated. If the `--save-id` flag"*
                   "is set, an additional .hdf5 file is generated. See `--save-ic` and `--pval-threshold`."
            required = true
        "--save-ic"
            help = "If the influence curves also need to be stored"
            default = false
            arg_type = Bool
            action = :store_true
        "--pval-threshold"
            help = "Only those parameters passing the threshold will have their influence curve saved."
            default = 1.
            arg_type = Float64
        "--verbosity", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

tmle_estimation(parsed_args)


